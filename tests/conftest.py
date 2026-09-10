import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--run-training", action="store_true", default=False, help="Explicitly allow tests that update model weights")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-training"):
        skip = pytest.mark.skip(reason="Training is disabled. Explicit --run-training is required.")
        for item in items:
            if "training" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(7)
