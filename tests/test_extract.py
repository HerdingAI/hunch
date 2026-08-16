import zipfile

import hunch.extract as extract
from hunch.config import Config

CFG = Config()


def test_plain_text_roundtrip(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("the quick brown fox")
    text, err = extract.extract_document(str(f), "txt", f.stat().st_size, CFG)
    assert "quick brown fox" in text
    assert err == ""


def test_nul_bytes_are_stripped(tmp_path):
    # Real logs and `strings` output carry NULs; they must never reach SQLite.
    f = tmp_path / "a.log"
    f.write_bytes(b"before\x00after")
    text, err = extract.extract_document(str(f), "log", f.stat().st_size, CFG)
    assert "\x00" not in text
    assert "before" in text and "after" in text


def test_docx_container_is_read(tmp_path):
    f = tmp_path / "a.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml",
                   "<w:p><w:t>contract terms herein</w:t></w:p>")
    text, err = extract.extract_document(str(f), "docx", f.stat().st_size, CFG)
    assert "contract terms herein" in text


def test_docx_container_with_oversized_member_is_capped_not_crashed(tmp_path):
    # A crafted member that decompresses far past MAX_ZIP_MEMBER_BYTES must
    # be skipped rather than fully read into memory (a "zip bomb" guard).
    f = tmp_path / "bomb.docx"
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "a" * (extract.MAX_ZIP_MEMBER_BYTES + 1))
    text, err = extract.extract_document(str(f), "docx", f.stat().st_size, CFG)
    assert text == ""          # the only member present was skipped, not read
    assert "no readable parts" in err


def test_ole2_file_named_docx_is_salvaged(tmp_path):
    # A .docx that is not a zip is almost always a legacy binary Office file
    # with a modern extension. Salvage it instead of discarding it.
    f = tmp_path / "legacy.docx"
    f.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 40 + b"QuarterlyReportText" + b"\x00" * 40)
    text, err = extract.extract_document(str(f), "docx", f.stat().st_size, CFG)
    assert "QuarterlyReportText" in text
    assert err == ""


def test_email_puts_headers_first(tmp_path):
    f = tmp_path / "m.eml"
    f.write_text(
        "From: alice@example.com\r\nTo: bob@example.com\r\n"
        "Subject: Roof repair quote\r\n\r\nThe quote is attached.\r\n")
    text, err = extract.extract_document(str(f), "eml", f.stat().st_size, CFG)
    # Headers lead so they survive the embedding truncation window.
    assert text.index("Roof repair quote") < text.index("The quote is attached")


def test_unparseable_type_returns_error_not_exception(tmp_path):
    f = tmp_path / "a.zzz"
    f.write_text("payload")
    text, err = extract.extract_document(str(f), "zzz", f.stat().st_size, CFG)
    assert text == ""
    assert err != ""


def test_child_timeout_is_reported(tmp_path):
    text, err = extract.run_child(["sleep", "5"], timeout=1)
    assert text == ""
    assert "timeout" in err.lower()
