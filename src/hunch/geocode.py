"""Offline reverse geocoding over a bundled GeoNames extract.

No network call, which keeps the privacy promise intact, and no LGPL
dependency, which keeps the Apache-2.0 licence clean.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "cities.npz"


@functools.lru_cache(maxsize=1)
def _cities():
    if not DATA.exists():
        return None
    with np.load(DATA, allow_pickle=True) as d:
        return d["lat"], d["lon"], d["name"]


def nearest_place(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    data = _cities()
    if data is None:
        return None
    lats, lons, names = data

    # Equirectangular approximation: exact enough to pick the nearest city and
    # far cheaper than haversine across the dataset. The raw longitude
    # difference is wrapped into [-180, 180] before use -- without this, a
    # point at 179.9 and a city at -179.9 (physically ~20km apart, just on
    # either side of the antimeridian) compute a ~360-degree difference
    # instead of the true ~0.2-degree one, mislabeling photos taken near the
    # dateline (Fiji, Tonga, Kiribati, the Aleutians, far-eastern Russia)
    # with a city from a different country entirely.
    lat_r = math.radians(lat)
    dlon = lons - lon
    dlon = (dlon + 180) % 360 - 180
    dx = dlon * math.cos(lat_r)
    dy = lats - lat
    idx = int(np.argmin(dx * dx + dy * dy))
    return str(names[idx])
