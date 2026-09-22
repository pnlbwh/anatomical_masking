"""Stdlib-only source provenance shared by training and notebook staging."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Every local module whose implementation defines training/resume semantics.
# Keep explicit: unrelated evaluation scripts must not invalidate a full resume.
TRAINING_SOURCE_FILES = (
    'train.py',
    'training/__init__.py',
    'training/engine.py',
    'training/config.py',
    'training/data.py',
    'training/checkpoints.py',
    'training/runtime.py',
    'training/provenance.py',
    'imaging/__init__.py',
    'imaging/geometry.py',
    'imaging/normalization.py',
    'imaging/metadata.py',
    'models/__init__.py',
    'models/architectures.py',
    'models/losses.py',
    'configuration/__init__.py',
    'configuration/presets.py',
)


def training_code_fingerprint(root) -> str:
    """Hash relative names and bytes; fail if any required runtime source is missing."""
    root = Path(root).resolve()
    files = {root / name for name in TRAINING_SOURCE_FILES}
    augmentation = root / "augmentations"
    files.update(augmentation.rglob("*.py"))
    files.update(augmentation.rglob("*.json"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"cannot fingerprint training source {path}: {exc}") from exc
        digest.update(b"\0")
    return digest.hexdigest()
