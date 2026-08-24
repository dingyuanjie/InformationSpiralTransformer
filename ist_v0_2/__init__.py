"""IST v0.2 experimental hierarchical internal Memory package."""

import sys
from pathlib import Path

# v0.1 intentionally keeps its script-style absolute imports. Add this isolated
# package directory without touching the sibling v0.1 installation.
_PACKAGE = str(Path(__file__).resolve().parent)
if _PACKAGE not in sys.path:
    sys.path.insert(0, _PACKAGE)

from config import HierarchicalMemoryConfig
from hierarchical_model import HierarchicalInformationSpiralTransformer, transfer_v0_1_weights
from model import InformationSpiralTransformer, build_model

__all__ = ["InformationSpiralTransformer", "HierarchicalInformationSpiralTransformer",
           "HierarchicalMemoryConfig", "build_model", "transfer_v0_1_weights"]
