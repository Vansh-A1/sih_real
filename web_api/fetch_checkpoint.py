"""Download the user's pinned checkpoint and verify it before installation."""
from pathlib import Path
import hashlib
import os
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5663e7e14479d6627b8d5e14d7e507302c73f5be"
URL = f"https://raw.githubusercontent.com/Vansh-A1/sih_real/{REVISION}/last.pt"
SHA256 = "df4eb605c0b795df9f339faa5c93af9c01329984e7c0d14b6392a4254da2bfa0"


def main():
    destination = ROOT / "last.pt"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
            print("The verified checkpoint is already installed.")
            return
        raise SystemExit("last.pt already exists with different contents. Set REZX_CHECKPOINT to use it, or move it before downloading the pinned checkpoint.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=ROOT, prefix=".rezx-checkpoint-", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Checkpoint checksum mismatch; no checkpoint was installed.")
        os.replace(temporary, destination)
        print(f"Installed verified checkpoint: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
