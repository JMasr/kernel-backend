from abc import ABC
from typing import ClassVar


class EmbedderPort(ABC):
    """Port for watermark embedding. Concrete implementations live in engine/."""

    signal_id: ClassVar[str]
