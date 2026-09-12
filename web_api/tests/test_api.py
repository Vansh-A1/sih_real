"""API contract and real-checkpoint inference checks. Never runs training."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import time
import zipfile

from affine import Affine
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.io import MemoryFile

import inference
from web_api.app import create_app, MAX_BYTES

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "last.pt"


@pytest.fixture(scope="module")
def client():
    if not CHECKPOINT.is_file():
        pytest.skip("Run python -m web_api.fetch_checkpoint for real-checkpoint API tests")
    with TestClient(create_app(CHECKPOINT, "cpu")) as connection:
        yield connection


def npy_bytes(array):
    handle = BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def upload(client, array, **settings):
    return client.post("/api/images", files={"file": ("scene.npy", npy_bytes(array), "application/octet-stream")}, data=settings)


def finish(client, job):
    end = time.monotonic() + 30
    while time.monotonic() < end:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] in ("complete", "failed", "cancelled"):
            return state
        time.sleep(.02)
    pytest.fail("Inference did not finish within 30 seconds")


def remove(client, image):
    assert client.delete(f"/api/images/{image['id']}").status_code == 204


def test_real_checkpoint_inference_and_shared_display(client):
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    health = client.get("/api/health").json()
    assert health["status"] == "ready" and health["step"] == 10000 and health["ema"] is True
    lr = np.random.default_rng(7).uniform(.1, .35, (4, 12, 16)).astype("float32")
    response = upload(client, lr)
    assert response.status_code == 201, response.text
    record = response.json()
    job = client.post("/api/jobs", json={"image_id": record["id"]}).json()
    state = finish(client, job)
    assert state["status"] == "complete", state
    assert state["progress"] == 100
    assert (state["result"]["width"], state["result"]["height"]) == (64, 48)
    archive = zipfile.ZipFile(BytesIO(client.get(state["result"]["download_url"]).content))
    assert set(archive.namelist()) == {"sr.npy", "sr.png", "inference.json"}
    sr = np.load(BytesIO(archive.read("sr.npy")), allow_pickle=False)
    model = client.app.state.service.model
    expected = inference.predict_sr(model, lr, np.ones_like(lr), "cpu", 128, 16, 16)
    np.testing.assert_allclose(sr, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(sr, np.repeat(np.repeat(lr, 4, 1), 4, 2))
    limits = inference.display_limits(lr[[2, 1, 0]])
    expected_rgb = np.rint(inference.rgb_preview(sr[[2, 1, 0]], limits) * 255).astype("uint8")
    np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(archive.read("sr.png")))), expected_rgb)
    report = json.loads(archive.read("inference.json"))
    assert report["checkpoint_sha256"] == digest and not report["reference_used_for_inference"]
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == digest
    remove(client, record)


def test_geotiff_footprint_and_nodata(client):
    transform = Affine(10, 2, 500000, 1, -10, 3000000)
    cube = np.random.default_rng(2).uniform(.1, .3, (4, 9, 11)).astype("float32")
    cube[:, 0, 0] = -9999
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=11, height=9, count=4, dtype="float32", crs="EPSG:32643", transform=transform, nodata=-9999) as dst:
            dst.write(cube)
            dst.descriptions = ("B2", "B3", "B4", "B8")
        content = memory.read()
    response = client.post("/api/images", files={"file": ("scene.tif", content, "image/tiff")})
    assert response.status_code == 201, response.text
    record = response.json()
    state = finish(client, client.post("/api/jobs", json={"image_id": record["id"]}).json())
    assert state["status"] == "complete", state
    archive = zipfile.ZipFile(BytesIO(client.get(state["result"]["download_url"]).content))
    with MemoryFile(archive.read("sr.tif")) as memory, memory.open() as dataset:
        assert dataset.crs.to_epsg() == 32643
        assert dataset.transform == transform @ Affine.scale(.25, .25)
        assert dataset.descriptions == ("B2", "B3", "B4", "B8")
        assert dataset.width == 44 and dataset.height == 36
        assert np.isnan(dataset.read()[:, :4, :4]).all()
    remove(client, record)


def test_explicit_decoding_and_hwc(client):
    cube = np.zeros((5, 7, 4), dtype="uint16")
    cube[:] = [100, 200, 300, 400]
    assert upload(client, cube, layout="HWC").status_code == 422
    response = upload(client, cube, layout="HWC", bands="3,2,1,4", scale="0.001", offset="-0.05")
    assert response.status_code == 201, response.text
    record = response.json()
    lr = np.load(client.app.state.service.images[record["id"]].directory / "lr.npy")
    np.testing.assert_allclose(lr[:, 2, 2], [.25, .15, .05, .35], atol=1e-6)
    remove(client, record)


@pytest.mark.parametrize("array,settings", [
    (np.ones((3, 8, 8), dtype="float32"), {}),
    (np.ones((4, 2, 1025), dtype="float32"), {}),
    (np.full((4, 8, 8), np.nan, dtype="float32"), {}),
    (np.ones((4, 8, 8), dtype="float32"), {"bands": "1,2,3,3"}),
    (np.ones((4, 8, 8), dtype="float32"), {"scale": "nan"}),
    (np.ones((4, 8, 8), dtype="float32"), {"layout": "guess"}),
    (np.ones((4, 8, 8), dtype="complex64"), {}),
])
def test_invalid_inputs(client, array, settings):
    before = len(client.app.state.service.images)
    assert upload(client, array, **settings).status_code == 422
    assert len(client.app.state.service.images) == before


def test_invalid_file_types_and_body_size(client):
    for name, data in [("photo.png", b"png"), ("bad.npy", b"invalid"), ("bad.tif", b"<VRTDataset/>")]:
        assert client.post("/api/images", files={"file": (name, data)}).status_code == 422
    assert client.post("/api/images", content=b"", headers={"content-length": str(MAX_BYTES * 2)}).status_code == 413


def test_cancel_queued_job_and_retry(client):
    import threading
    service = client.app.state.service
    gate = threading.Event()
    service.executor.submit(lambda: gate.wait(5))
    record = client.post("/api/images/sample").json()
    try:
        job = client.post("/api/jobs", json={"image_id": record["id"]}).json()
        assert client.post("/api/jobs", json={"image_id": record["id"]}).json()["id"] == job["id"]
        assert client.delete(f"/api/images/{record['id']}").status_code == 409
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 202
    finally:
        gate.set()
    assert finish(client, job)["status"] == "cancelled"
    retried = client.post("/api/jobs", json={"image_id": record["id"]}).json()
    assert retried["id"] != job["id"]
    assert finish(client, retried)["status"] == "complete"
    assert service.model._forward_hooks == {} and service.model._forward_pre_hooks == {}
    remove(client, record)


def test_missing_checkpoint_and_unknown_results(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt", "cpu")) as connection:
        assert connection.get("/api/health").json()["status"] == "unavailable"
        assert connection.post("/api/jobs", json={"image_id": "none"}).status_code == 503
        assert connection.get("/api/jobs/none").status_code == 404


def test_cancel_during_real_model_forward(client):
    import threading
    service = client.app.state.service
    entered, release = threading.Event(), threading.Event()

    def hold_completed_tile(*_):
        entered.set()
        release.wait(5)

    hook = service.model.register_forward_hook(hold_completed_tile)
    record = client.post('/api/images/sample').json()
    try:
        job = client.post('/api/jobs', json={'image_id': record['id']}).json()
        assert entered.wait(5), 'Model did not run'
        state = client.get(f"/api/jobs/{job['id']}").json()
        assert state['status'] == 'processing' and state['progress'] < 100
        client.delete(f"/api/jobs/{job['id']}")
    finally:
        release.set()
        hook.remove()
    assert finish(client, job)['status'] == 'cancelled'
    assert service.model._forward_hooks == {} and service.model._forward_pre_hooks == {}
    remove(client, record)
