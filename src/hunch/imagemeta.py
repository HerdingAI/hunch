"""The cheap tier of image understanding.

Roughly 0.2s per image versus ~5s for a vision model, and it already answers a
large share of real queries -- "photos from the italy trip", "scanned receipts
from 2019", "screenshots about invoices". A vision description upgrades this
record later; it does not replace it.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import geocode
from .config import Config
from .extract import OCR_TIMEOUT, clean_text, run_child

_GPS_TAG = 34853
_DATETIME_TAGS = (36867, 306)   # DateTimeOriginal, DateTime
_MODEL_TAG = 272


def _ratio(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gps(exif) -> tuple[float | None, float | None]:
    try:
        gps = exif.get_ifd(_GPS_TAG)
    except Exception:                              # noqa: BLE001
        return None, None
    if not gps:
        return None, None
    try:
        lat_dms, lat_ref = gps.get(2), gps.get(1)
        lon_dms, lon_ref = gps.get(4), gps.get(3)
        if not lat_dms or not lon_dms:
            return None, None
        lat = _ratio(lat_dms[0]) + _ratio(lat_dms[1]) / 60 + _ratio(lat_dms[2]) / 3600
        lon = _ratio(lon_dms[0]) + _ratio(lon_dms[1]) / 60 + _ratio(lon_dms[2]) / 3600
        if str(lat_ref).upper().startswith("S"):
            lat = -lat
        if str(lon_ref).upper().startswith("W"):
            lon = -lon
        return lat, lon
    except Exception:                              # noqa: BLE001
        return None, None


def exif_summary(path: str) -> dict:
    """Pillow's built-in EXIF reader -- permissive licence, no extra dependency."""
    out: dict = {}
    try:
        from PIL import Image
        with Image.open(path) as img:
            out["width"], out["height"] = img.size
            exif = img.getexif()
        if exif:
            for tag in _DATETIME_TAGS:
                if exif.get(tag):
                    out["taken"] = str(exif.get(tag))
                    break
            if exif.get(_MODEL_TAG):
                out["camera"] = str(exif.get(_MODEL_TAG)).strip()
            lat, lon = _gps(exif)
            if lat is not None:
                place = geocode.nearest_place(lat, lon)
                if place:
                    out["place"] = place
    except Exception:                              # noqa: BLE001
        pass          # a photo with unreadable EXIF still has a path and OCR
    return out


def ocr(path: str, cfg: Config) -> str:
    text, _ = run_child(["tesseract", path, "stdout", "-l", cfg.ocr_languages],
                        OCR_TIMEOUT)
    return text.strip()


def describe(path: str, cfg: Config) -> str:
    meta = exif_summary(path)
    p = Path(path)
    parts = [f"file: {os.path.join(p.parent.name, p.name)}"]
    if meta.get("taken"):
        parts.append(f"taken: {meta['taken']}")
    if meta.get("place"):
        parts.append(f"place: {meta['place']}")
    if meta.get("camera"):
        parts.append(f"camera: {meta['camera']}")
    if meta.get("width"):
        parts.append(f"size: {meta['width']}x{meta['height']}")
    found = ocr(path, cfg)
    if found:
        parts.append(f"text in image: {found}")
    return clean_text("\n".join(parts))
