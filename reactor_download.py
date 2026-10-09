import os
import tempfile
import urllib.request

from tqdm import tqdm


def download(url, path, name):
    """Publish a model file only after a complete, nonempty transfer."""
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            header = response.headers.get("Content-Length")
            total = int(header) if header is not None else None
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=f".{os.path.basename(destination)}.",
                suffix=".part", delete=False,
            ) as output:
                temporary_path = output.name
                with tqdm(
                    total=total, desc=f"[ReActor] Downloading {name} to {path}",
                    unit="B", unit_scale=True, unit_divisor=1024,
                ) as progress:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
                received = output.tell()
                if received == 0 or (total is not None and received != total):
                    raise OSError(
                        f"Incomplete download of {name}: received {received} bytes"
                        f", expected {total if total is not None else 'a nonempty file'}."
                    )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.remove(temporary_path)
