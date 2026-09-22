"""Checkpoint preprocessing fingerprints, independent of augmentation code."""

# Keep legacy key names so existing checkpoint hashes remain valid.
_PREPROC_HASH_KEYS = ("model_type", "target_shape", "patch_size", "conform_mm",
                      "canonical_orientation", "zscore_in_mask", "artifact_names", "model_kwargs")


def preproc_config_hash(config) -> str:
    """Hash preprocessing settings consistently across Python and JSON round-trips."""
    import hashlib
    import json
    sub = {}
    for k in _PREPROC_HASH_KEYS:
        v = (config or {}).get(k)
        sub[k] = list(v) if isinstance(v, tuple) else v      # tuple/list normalize (JSON emits lists)
    blob = json.dumps(sub, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
