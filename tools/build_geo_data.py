"""Build the bundled offline city dataset from GeoNames (CC BY 4.0).

Both PyPI reverse-geocoding packages are LGPL, which is awkward under
Apache-2.0. cities15000 is ~25k rows, so nearest-neighbour over it is a
trivial numpy operation and needs no third-party geocoder.

Usage: python tools/build_geo_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://download.geonames.org/export/dump/cities15000.zip"
OUT = Path(__file__).resolve().parent.parent / "src" / "hunch" / "data" / "cities.npz"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read("cities15000.txt").decode("utf-8").splitlines()

    lats, lons, names = [], [], []
    for line in lines:
        f = line.split("\t")
        # 1=name, 4=lat, 5=lon, 8=country code, 10=admin1 code
        lats.append(float(f[4]))
        lons.append(float(f[5]))
        names.append(f"{f[1]}, {f[8]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        lat=np.asarray(lats, dtype=np.float32),
        lon=np.asarray(lons, dtype=np.float32),
        name=np.asarray(names, dtype=object),
    )
    print(f"wrote {OUT} with {len(names):,} cities")


if __name__ == "__main__":
    main()
