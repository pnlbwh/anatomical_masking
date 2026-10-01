"""Mask a directory with the condensed pipeline; see --help for filters/options."""
from pathlib import Path
import sys

# Support both `python -m anatomical_masking.mask_directory` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anatomical_masking.inference.directory import MaskDirectory, main

__all__ = ["MaskDirectory", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
