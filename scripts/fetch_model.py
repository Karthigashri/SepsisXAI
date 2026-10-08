from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_MODEL_BYTES = 100_000_000
CHUNK_BYTES = 1024 * 1024


def fetch_model(url: str, expected_sha256: str, destination: Path) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("SEPSISPULSE_MODEL_URL must be an HTTPS URL.")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF"
        for character in expected_sha256
    ):
        raise ValueError("SEPSISPULSE_MODEL_SHA256 must be a 64-digit hex digest.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "SepsisPulse/0.1"})
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with urlopen(request, timeout=30) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https":
                raise ValueError("Model download redirected to a non-HTTPS URL.")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_MODEL_BYTES:
                        raise ValueError(
                            f"Model artifact exceeds {MAX_MODEL_BYTES} bytes."
                        )
                    digest.update(chunk)
                    temporary_file.write(chunk)

        if total_bytes == 0:
            raise ValueError("Downloaded model artifact is empty.")
        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ValueError("Downloaded model artifact SHA-256 did not match.")
        if temporary_path is None:
            raise RuntimeError("The model artifact download was not created.")
        from sepsispulse.inference import load_model_bundle

        load_model_bundle(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    url = os.environ.get("SEPSISPULSE_MODEL_URL", "")
    expected_sha256 = os.environ.get("SEPSISPULSE_MODEL_SHA256", "")
    destination = Path(
        os.environ.get(
            "SEPSISPULSE_MODEL_PATH", "artifacts/sepsispulse.joblib"
        )
    )
    if not url or not expected_sha256:
        raise RuntimeError(
            "Set SEPSISPULSE_MODEL_URL and SEPSISPULSE_MODEL_SHA256 to the "
            "trusted trained artifact and its digest before deployment."
        )
    fetch_model(url, expected_sha256, destination)
    print(f"Verified model artifact installed at {destination}")


if __name__ == "__main__":
    main()
