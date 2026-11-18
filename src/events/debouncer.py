from typing import Dict, Tuple
from src.schemas import EventType


class EventDebouncer:
    """Prevents duplicate alerts per (zone_name, event_type) during cooldown window."""

    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: Dict[Tuple[str, str], float] = {}

    def should_trigger(self, zone_name: str, event_type: EventType, current_ts: float) -> bool:
        key = (zone_name, event_type.value)
        last_ts = self._last_fired.get(key)
        if last_ts is not None and (current_ts - last_ts) < self.cooldown_seconds:
            return False
        self._last_fired[key] = current_ts
        return True
