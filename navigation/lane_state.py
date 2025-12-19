"""Shared lane detection state container."""

from typing import Optional, Dict, Any


class LaneState:
    """Shared lane detection state. Updated by perception, read by navigation."""

    def __init__(self) -> None:
        self._detection: Optional[Dict[str, Any]] = None
    
    @property
    def valid(self) -> bool:
        return self._detection is not None and self._detection['valid']
    
    @property
    def confidence(self) -> float:
        return self._detection['confidence'] if self._detection else 0.0
    
    def update(self, detection_result: dict) -> None:
        """Update state from lane detector output dict."""
        self._detection = detection_result
    
    def reset(self) -> None:
        """Reset to invalid state."""
        self._detection = None
    
    def to_dict(self) -> Optional[dict]:
        """Get the raw detection dict."""
        return self._detection
