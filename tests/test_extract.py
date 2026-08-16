import struct
import time
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


def test_docx_container_with_spoofed_member_size_does_not_bypass_the_cap(tmp_path):
    # A zip member's declared uncompressed size (in its local file header and
    # central directory record) is attacker-supplied metadata, not a
    # measured quantity. This builds a member that DECLARES a tiny size
    # while its real compressed payload decompresses to well past the cap,
    # then patches both size fields directly -- the same spoof a crafted
    # malicious .docx in a Downloads folder could carry.
    #
    # A correctness-only assertion (text == "") is NOT sufficient here and
    # would pass even on the vulnerable code: highly-compressible "a" * N
    # content still fails its own CRC-32 check once decompressed, whether
    # that decompression was bounded or not, so both the buggy and the fixed
    # implementation land on an identical empty result. The property that
    # actually distinguishes them is resource cost: the vulnerable path
    # decompresses the full real payload (hundreds of ms, scaling with real
    # content size) before the CRC mismatch is caught; the fixed path is
    # gated by the tiny spoofed declared size and stays near-instant
    # regardless of real content size. 300MB/150ms were chosen empirically
    # and verified against both implementations directly: on the reference
    # dev machine the fixed path measured ~23ms at this size (6.5x margin
    # below the threshold) and a deliberately-reverted vulnerable path
    # measured ~416ms (2.8x margin above it) -- comfortable in both
    # directions against CI timing variance.
    f = tmp_path / "spoofed.docx"
    real_size = 300 * 1024 * 1024
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "a" * real_size)

    data = bytearray(f.read_bytes())
    spoofed_size = 1000
    lfh_off = data.find(b"PK\x03\x04")           # local file header
    struct.pack_into("<I", data, lfh_off + 22, spoofed_size)
    cdh_off = data.find(b"PK\x01\x02")           # central directory record
    struct.pack_into("<I", data, cdh_off + 24, spoofed_size)
    f.write_bytes(data)

    with zipfile.ZipFile(f) as zf:
        assert zf.getinfo("word/document.xml").file_size == spoofed_size

    started = time.time()
    text, err = extract.extract_document(str(f), "docx", f.stat().st_size, CFG)
    elapsed = time.time() - started

    assert text == ""          # must not have decompressed and returned the payload
    assert elapsed < 0.15, (
        f"took {elapsed:.3f}s -- the size cap should be gated by bytes "
        "actually read, not the zip's declared (spoofable) member size")
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
