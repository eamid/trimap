from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from .torch_trimap import TorchTRIMAP, trimap_explicit_grad, trimap_loss

if TYPE_CHECKING:
    from .trimap_ import TRIMAP

try:
    __version__ = version("trimap")
except PackageNotFoundError:
    __version__ = "unknown"


def __getattr__(name: str) -> Any:
    """Load the Numba/Annoy implementation only when legacy TRIMAP is used."""
    if name == "TRIMAP":
        from .trimap_ import TRIMAP

        globals()[name] = TRIMAP
        return TRIMAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TRIMAP", "TorchTRIMAP", "trimap_explicit_grad", "trimap_loss"]
