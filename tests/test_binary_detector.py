"""Tests for BinaryDetector — extension and content-based binary file detection."""

import os
import tempfile

import pytest

from engine.tools.builtin._utils.binary import BinaryDetector, BINARY_EXTENSIONS


class TestBinaryExtensionDetection:
    """Tests for extension-based binary detection."""

    def test_exe_is_binary(self):
        assert BinaryDetector.is_binary_extension("program.exe")

    def test_zip_is_binary(self):
        assert BinaryDetector.is_binary_extension("archive.zip")

    def test_png_is_binary(self):
        assert BinaryDetector.is_binary_extension("image.png")

    def test_jpg_is_binary(self):
        assert BinaryDetector.is_binary_extension("photo.jpg")

    def test_pdf_is_binary(self):
        assert BinaryDetector.is_binary_extension("document.pdf")

    def test_pyc_is_binary(self):
        assert BinaryDetector.is_binary_extension("module.pyc")

    def test_mp4_is_binary(self):
        assert BinaryDetector.is_binary_extension("video.mp4")

    def test_py_is_not_binary(self):
        assert not BinaryDetector.is_binary_extension("script.py")

    def test_txt_is_not_binary(self):
        assert not BinaryDetector.is_binary_extension("notes.txt")

    def test_md_is_not_binary(self):
        assert not BinaryDetector.is_binary_extension("readme.md")

    def test_json_is_not_binary(self):
        assert not BinaryDetector.is_binary_extension("data.json")

    def test_case_insensitive(self):
        assert BinaryDetector.is_binary_extension("FILE.EXE")
        assert BinaryDetector.is_binary_extension("File.Png")

    def test_no_extension_is_not_binary(self):
        assert not BinaryDetector.is_binary_extension("Makefile")


class TestBinaryContentDetection:
    """Tests for content-based binary detection using temp files."""

    def test_null_byte_file_is_binary(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b"hello\x00world")
        assert BinaryDetector.is_binary_content(str(f))

    def test_text_file_is_not_binary(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, World!\nThis is a text file.\n")
        assert not BinaryDetector.is_binary_content(str(f))

    def test_empty_file_is_not_binary(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert not BinaryDetector.is_binary_content(str(f))

    def test_nonexistent_file_is_not_binary(self):
        # Non-existent files should return False, not raise
        assert not BinaryDetector.is_binary_content("/nonexistent/path/file.txt")

    def test_utf8_bom_not_binary(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfHello UTF-8 with BOM")
        assert not BinaryDetector.is_binary_content(str(f))

    def test_mostly_nonprintable_is_binary(self, tmp_path):
        # Create a file with >30% non-printable characters (excluding tab/newline/CR)
        chunk = bytes(range(1, 32)) * 50  # Lots of control characters
        f = tmp_path / "control.dat"
        f.write_bytes(chunk)
        assert BinaryDetector.is_binary_content(str(f))


class TestBinaryCombinedDetection:
    """Tests for the combined is_binary() method."""

    def test_binary_by_extension(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_text("actually text")
        # Extension check first — returns True even if content is text
        assert BinaryDetector.is_binary(str(f))

    def test_binary_by_content(self, tmp_path):
        # No extension — falls through to content check
        f = tmp_path / "binary_data"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        assert BinaryDetector.is_binary(str(f))

    def test_text_file_not_binary(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("print('hello')")
        assert not BinaryDetector.is_binary(str(f))


class TestBinaryExtensionsSet:
    """Tests for the BINARY_EXTENSIONS constant."""

    def test_extensions_is_frozenset(self):
        assert isinstance(BINARY_EXTENSIONS, frozenset)

    def test_contains_common_extensions(self):
        for ext in [".exe", ".zip", ".png", ".jpg", ".pdf", ".mp3", ".mp4", ".pyc", ".dll", ".so"]:
            assert ext in BINARY_EXTENSIONS, f"{ext} should be in BINARY_EXTENSIONS"

    def test_has_50_plus_extensions(self):
        assert len(BINARY_EXTENSIONS) >= 50
