"""Checked extraction into a fresh temporary tree. Never overwrite existing data."""
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from scaf.runtime import sha256
from .catalog import confined


def extract_sequence(spec, data_root):
    destination = confined(data_root, spec["path"])
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archives = []
    for item in spec["archives"]:
        archive = confined(data_root, "downloads/"+item["file"])
        if sha256(archive) != item["sha256"]:
            raise ValueError(f"archive SHA256 mismatch: {archive.name}")
        archives.append(archive)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        for archive in archives:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as handle:
                    for info in handle.infolist():
                        confined(temporary, info.filename)
                        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                            raise ValueError("archive symbolic links are not allowed")
                        target = confined(temporary, info.filename)
                        if info.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with handle.open(info) as source, target.open("xb") as output:
                                shutil.copyfileobj(source, output)  # validates CRC while reading
            else:
                with tarfile.open(archive, "r:gz") as handle:
                    for member in handle:
                        target = confined(temporary, member.name)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with handle.extractfile(member) as source, target.open("xb") as output:
                                shutil.copyfileobj(source, output)
                        else:
                            raise ValueError("only regular archive files/directories are allowed")
        expected = temporary / destination.name
        if not expected.is_dir() or sorted(p.name for p in temporary.iterdir()) != [destination.name]:
            raise ValueError("unexpected archive top-level layout")
        expected.rename(destination)
    return destination
