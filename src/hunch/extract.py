"""Turn a file into text.

Formats parsed by native C libraries with a history of memory-corruption
bugs on malformed input (pdftotext, tesseract, ffmpeg) are handed to an
external binary under a hard timeout rather than parsed in-process, so a
crash or a hang costs one file, not the run. zipfile is pure Python and
isn't exposed to that crash class, so ZIP-based containers (docx/pptx/xlsx/
odt/epub) are read in-process instead -- bounded by MAX_ZIP_MEMBER_BYTES and
MAX_ZIP_TOTAL_BYTES so a crafted small archive can't exhaust worker memory.
"""
from __future__ import annotations

import os
import re
import subprocess
import zipfile
from pathlib import Path

from .config import Config

PDF_TIMEOUT = 120
OCR_TIMEOUT = 180
FFMPEG_TIMEOUT = 900

MAX_DOC_BYTES = 200 * 1024 * 1024
# Every other risky format here goes through an external binary under a hard
# timeout so a crash or hang costs one file, not the run. zipfile is pure
# Python and can't corrupt memory the way poppler/tesseract can, but reading
# an unbounded member gives a crafted small zip (a "zip bomb") a way to
# exhaust worker memory in-process -- these caps close that gap.
MAX_ZIP_MEMBER_BYTES = 20 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 50 * 1024 * 1024
# Below this many characters a PDF has no usable text layer -- it is either a
# vector graphic or, far more importantly, a scan.
MIN_PDF_TEXT = 50
PDF_OCR_PAGES = 5
# Title, contents and preface are what identify an oversized book.
BIG_PDF_PAGES = 40

TEXTISH = {"txt", "md", "markdown", "csv", "tsv", "json", "xml",
           "yaml", "yml", "log", "tex", "org", "html", "htm"}
ZIP_XML = {"docx", "pptx", "xlsx", "odt", "epub"}
LEGACY_BINARY = {"doc", "ppt", "xls", "rtf"}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return (value.encode("utf-8", "replace").decode("utf-8", "replace")
            .replace("\x00", ""))


def run_child(cmd: list[str], timeout: int) -> tuple[str, str]:
    """Run an external parser. Returns (stdout, error)."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout}s: {cmd[0]}"
    except FileNotFoundError:
        return "", f"missing binary: {cmd[0]} (install it with apt)"
    except Exception as exc:                       # noqa: BLE001
        return "", f"{cmd[0]} failed: {exc}"
    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0 and not out.strip():
        tail = proc.stderr.decode("utf-8", "replace").strip()[-200:]
        return "", f"{cmd[0]} exit {proc.returncode}: {tail}"
    return out, ""


def _plain_text(path: str, size: int) -> tuple[str, str]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(min(size, MAX_DOC_BYTES))
        return raw.decode("utf-8", "replace"), ""
    except OSError as exc:
        return "", f"read failed: {exc}"


def _zip_xml(path: str, ext: str) -> tuple[str, str]:
    wanted = {"docx": ("word/document.xml",), "xlsx": ("xl/sharedStrings.xml",),
              "odt": ("content.xml",), "pptx": None, "epub": None}[ext]
    try:
        chunks = []
        total = 0
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if wanted is None:
                if ext == "pptx":
                    names = [n for n in names if n.startswith("ppt/slides/slide")
                             and n.endswith(".xml")]
                else:
                    names = [n for n in names if n.endswith((".xhtml", ".html", ".htm"))]
            else:
                names = [n for n in names if n in wanted]
            for name in names[:400]:
                try:
                    # Bound bytes actually read, not the member's *declared*
                    # size: zf.getinfo(name).file_size lives in the local
                    # file header / central directory as attacker-controlled
                    # metadata, not a measured quantity. A crafted member can
                    # declare a tiny size while its real payload decompresses
                    # to hundreds of MB -- zf.read() doesn't notice the lie
                    # until *after* fully decompressing (the eventual CRC
                    # check fails too late to bound memory). Reading through
                    # the file object with a capped read() closes that gap.
                    with zf.open(name) as mf:
                        data = mf.read(MAX_ZIP_MEMBER_BYTES + 1)
                    if len(data) > MAX_ZIP_MEMBER_BYTES:
                        continue      # one pathological member skips, doc doesn't
                    total += len(data)
                    if total > MAX_ZIP_TOTAL_BYTES:
                        break
                    chunks.append(data.decode("utf-8", "replace"))
                except Exception:                  # noqa: BLE001
                    continue
    except (zipfile.BadZipFile, OSError) as exc:
        return "", f"bad {ext} container: {exc}"
    if not chunks:
        return "", f"no readable parts in {ext}"
    xml = re.sub(r"<[^>]+>", " ", "\n".join(chunks))
    return re.sub(r"\s+", " ", xml).strip(), ""


def _pdf_ocr(path: str, cfg: Config, pages: int = PDF_OCR_PAGES) -> tuple[str, str]:
    """Rasterise the first pages and OCR them.

    Used only when the text layer is missing. Without this fallback every
    scanned document on the machine silently fails.
    """
    import glob
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "pg")
        _, err = run_child(["pdftoppm", "-r", "200", "-f", "1", "-l", str(pages),
                            "-png", path, prefix], PDF_TIMEOUT)
        images = sorted(glob.glob(prefix + "*.png"))
        if not images:
            return "", err or "could not rasterise pdf"
        out = []
        for image in images:
            text, _ = run_child(["tesseract", image, "stdout", "-l", cfg.ocr_languages],
                                OCR_TIMEOUT)
            if text.strip():
                out.append(text)
        return "\n".join(out), ""


def _email(path: str) -> tuple[str, str]:
    import email
    import email.policy

    try:
        with open(path, "rb") as fh:
            msg = email.message_from_binary_file(fh, policy=email.policy.default)
    except Exception as exc:                       # noqa: BLE001
        return "", f"unparseable email: {exc}"

    # Headers lead deliberately: sender, recipient and subject are what people
    # remember about an email, so they must land inside the embedding's
    # truncation window.
    head = [f"{f}: {msg.get(f)}" for f in ("From", "To", "Cc", "Subject", "Date")
            if msg.get(f)]
    body, html = "", ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_filename():
                    continue
                ctype = part.get_content_type()
                if ctype == "text/plain" and not body:
                    body = part.get_content()
                elif ctype == "text/html" and not html:
                    html = part.get_content()
        else:
            body = msg.get_content()
    except Exception:                              # noqa: BLE001
        pass
    if not body and html:
        body = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", "\n".join(head) + "\n\n" + (body or "")).strip()
    return (text, "") if text else ("", "email had no readable text")


def extract_document(path: str, ext: str, size: int, cfg: Config) -> tuple[str, str]:
    ext = (ext or "").lower().lstrip(".")

    if ext in TEXTISH:
        text, err = _plain_text(path, size)
        return clean_text(text), err

    if ext == "pdf":
        # No -q: poppler's syntax warnings on stderr are the only signal
        # distinguishing "corrupt / not really a PDF" from "valid but scanned".
        cmd = ["pdftotext", "-enc", "UTF-8"]
        if size > MAX_DOC_BYTES:
            # Oversized PDFs are usually scanned books -- real content worth
            # finding. Index the opening pages rather than refusing the file.
            cmd += ["-f", "1", "-l", str(BIG_PDF_PAGES)]
        cmd += [path, "-"]
        text, err = run_child(cmd, PDF_TIMEOUT)
        if len(text.strip()) < MIN_PDF_TEXT:
            ocr, ocr_err = _pdf_ocr(path, cfg)
            if ocr.strip():
                return clean_text(ocr), ""
            if not text.strip():
                return "", (err or ocr_err or "no text layer and OCR found nothing")
        return clean_text(text), err

    if ext in ("eml", "msg", "mbox"):
        text, err = _email(path)
        return clean_text(text), err

    if ext in ZIP_XML:
        text, err = _zip_xml(path, ext)
        if text.strip():
            return clean_text(text), ""
        # Only salvage via `strings` when the container itself failed to open
        # as a zip at all (the OLE2-with-a-modern-extension case). A file
        # that *did* open as a valid zip but had nothing extractable under
        # the size caps must not fall back to `strings` on the raw archive:
        # zip local/central-directory headers store member filenames as
        # literal uncompressed text, so `strings` on a legitimate-but-capped
        # container would "salvage" its own filenames as if they were
        # content, silently defeating the zip-bomb guard above.
        if err.startswith("bad "):
            salvaged, _ = run_child(["strings", "-n", "6", path], PDF_TIMEOUT)
            if salvaged.strip():
                return clean_text(salvaged), ""
        return "", err

    if ext in LEGACY_BINARY:
        text, err = run_child(["strings", "-n", "6", path], PDF_TIMEOUT)
        return clean_text(text), err

    return "", f"no extractor for .{ext}"
