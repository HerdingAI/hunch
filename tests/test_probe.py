from hunch.setup import probe


def test_capabilities_are_graded_not_pass_fail():
    caps = probe.Capabilities(has_gpu=False, vram_mb=0, ram_mb=8192,
                              free_disk_mb=50000, cpu_count=4,
                              has_tesseract=True, has_poppler=True,
                              has_ffmpeg=True)
    v = caps.verdict()
    # No GPU must never mean "images unsupported" -- OCR/EXIF/geocoding are
    # CPU-only and always on; only vision captioning needs a GPU.
    assert v["documents"] is True
    assert v["image_text"] is True
    assert v["photo_descriptions"] is False
    assert v["transcription"] is True
    assert "document" in v["summary"].lower()


def test_capable_machine_unlocks_everything():
    caps = probe.Capabilities(has_gpu=True, vram_mb=4096, ram_mb=16384,
                              free_disk_mb=80000, cpu_count=8,
                              has_tesseract=True, has_poppler=True,
                              has_ffmpeg=True)
    v = caps.verdict()
    assert all([v["documents"], v["image_text"], v["photo_descriptions"],
                v["transcription"]])
    assert "local" in v["summary"].lower()


def test_missing_tesseract_disables_image_text_only():
    caps = probe.Capabilities(has_gpu=True, vram_mb=4096, ram_mb=16384,
                              free_disk_mb=80000, cpu_count=8,
                              has_tesseract=False, has_poppler=True,
                              has_ffmpeg=True)
    v = caps.verdict()
    assert v["image_text"] is False
    assert v["photo_descriptions"] is True    # captioning needs no tesseract


def test_gpu_below_threshold_disables_only_photo_descriptions():
    caps = probe.Capabilities(has_gpu=True, vram_mb=2048, ram_mb=8192,
                              free_disk_mb=50000, cpu_count=4,
                              has_tesseract=True, has_poppler=True,
                              has_ffmpeg=True)
    v = caps.verdict()
    assert v["photo_descriptions"] is False
    assert v["image_text"] is True            # OCR still works without a strong GPU


def test_probe_runs_without_raising():
    caps = probe.probe()
    assert caps.cpu_count >= 1


def test_on_ac_power_true_when_mains_online(tmp_path, monkeypatch):
    ac = tmp_path / "AC"
    ac.mkdir()
    (ac / "type").write_text("Mains\n")
    (ac / "online").write_text("1\n")
    monkeypatch.setattr(probe, "_POWER_SUPPLY_DIR", tmp_path)
    assert probe.on_ac_power() is True


def test_on_ac_power_false_when_mains_offline(tmp_path, monkeypatch):
    ac = tmp_path / "AC"
    ac.mkdir()
    (ac / "type").write_text("Mains\n")
    (ac / "online").write_text("0\n")
    bat = tmp_path / "BAT0"
    bat.mkdir()
    (bat / "type").write_text("Battery\n")
    monkeypatch.setattr(probe, "_POWER_SUPPLY_DIR", tmp_path)
    assert probe.on_ac_power() is False


def test_on_ac_power_true_when_no_battery_hardware_at_all(tmp_path, monkeypatch):
    # A desktop with no Mains entry at all (nothing to be "unplugged" from)
    # must never be treated as perpetually on battery.
    monkeypatch.setattr(probe, "_POWER_SUPPLY_DIR", tmp_path)
    assert probe.on_ac_power() is True
