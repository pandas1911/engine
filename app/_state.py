"""Simple module-level state for concurrent request tracking."""

_is_streaming = False


def is_streaming() -> bool:
    return _is_streaming


def set_streaming(value: bool) -> None:
    global _is_streaming
    _is_streaming = value
