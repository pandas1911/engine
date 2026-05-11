"""Binary file detection — extension and content-based checks."""

from pathlib import Path

# Comprehensive binary extension blacklist (mirrors OpenCode's list)
BINARY_EXTENSIONS = frozenset({
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".iso", ".dmg", ".tgz",
    # Executables & libraries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".class", ".wasm",
    # Media (images)
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff", ".tif",
    ".svgz",
    # Audio/Video
    ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".avi", ".mkv", ".mov", ".wmv",
    # Documents (binary formats)
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Compiled/intermediate
    ".pyc", ".pyo", ".node", ".dex",
    # Other binary
    ".sqlite", ".db", ".lock",
})

_SAMPLE_SIZE = 8192  # Bytes to read for content detection
_NULL_BYTE_THRESHOLD = 0.30  # 30% non-printable = binary


class BinaryDetector:
    """Detect binary files via extension and content analysis."""

    @staticmethod
    def is_binary(path: str) -> bool:
        """Check if a file is binary using extension first, then content."""
        if BinaryDetector.is_binary_extension(path):
            return True
        return BinaryDetector.is_binary_content(path)

    @staticmethod
    def is_binary_extension(path: str) -> bool:
        """Check if file extension indicates a binary file."""
        return Path(path).suffix.lower() in BINARY_EXTENSIONS

    @staticmethod
    def is_binary_content(path: str) -> bool:
        """Check file content for binary indicators (null bytes, non-printable ratio).

        Reads only the first _SAMPLE_SIZE bytes for efficiency.
        Returns False for empty files (treat as text).
        """
        try:
            with open(path, "rb") as f:
                chunk = f.read(_SAMPLE_SIZE)
        except (OSError, IOError):
            return False

        if not chunk:
            return False  # Empty files are not binary

        # Null byte = definitely binary
        if b"\x00" in chunk:
            return True

        # Non-printable character ratio check
        non_printable = sum(
            1 for b in chunk
            if b < 32 and b not in (9, 10, 13)  # tab, newline, carriage return
        )
        return (non_printable / len(chunk)) > _NULL_BYTE_THRESHOLD
