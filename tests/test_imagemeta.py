import numpy as np
import pytest

import hunch.geocode as geocode
import hunch.imagemeta as imagemeta
from hunch.config import Config

CFG = Config()
PIL = pytest.importorskip("PIL")


def test_nearest_place_finds_a_known_city():
    # Florence, Italy
    place = geocode.nearest_place(43.7696, 11.2558)
    assert place is not None
    assert "Firenze" in place or "Florence" in place


def test_nearest_place_returns_none_for_invalid_coords():
    assert geocode.nearest_place(None, None) is None
    assert geocode.nearest_place(999.0, 999.0) is None


def test_nearest_place_handles_the_antimeridian():
    # Two points ~20km apart at this latitude, straddling the 180-degree
    # meridian, must resolve to the same nearby place -- not two different
    # countries hundreds of kilometers apart, which is what an unwrapped
    # longitude difference produces (raw ~360-degree gap instead of ~0.2).
    east = geocode.nearest_place(-17.7, 179.9)
    west = geocode.nearest_place(-17.7, -179.9)
    assert east is not None and west is not None
    assert east == west


def test_describe_includes_folder_path_even_with_no_exif(tmp_path):
    from PIL import Image
    d = tmp_path / "Trips" / "Italy 2019"
    d.mkdir(parents=True)
    img = d / "IMG_2033.jpg"
    Image.new("RGB", (64, 48), (128, 128, 128)).save(img)

    text = imagemeta.describe(str(img), CFG)
    # Folder path is often the most informative signal about a photo.
    assert "Italy 2019" in text
    assert "IMG_2033" in text


def test_describe_reports_dimensions(tmp_path):
    from PIL import Image
    img = tmp_path / "p.png"
    Image.new("RGB", (320, 200), (10, 20, 30)).save(img)
    text = imagemeta.describe(str(img), CFG)
    assert "320" in text and "200" in text
