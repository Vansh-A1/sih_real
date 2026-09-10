"""S2-EvidenceSR-4X. No downloads or training occur on import."""
__version__ = "0.1.0"

from .config import Config, load_config
from .models.network import EvidenceSR, SRResult

__all__ = ["Config", "load_config", "EvidenceSR", "SRResult"]
