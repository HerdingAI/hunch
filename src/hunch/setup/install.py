"""Install user-level integration. Nothing here needs root.

A GNOME Shell search provider is deliberately absent: Shell loads providers
with collectFromDatadirs('search-providers', false), so a provider must be
written to a system data dir, which would require privilege escalation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SERVICE = """[Unit]
Description=Hunch file indexing
After=default.target

[Service]
Type=oneshot
ExecStart={exe} index --scheduled
Nice=19
IOSchedulingClass=idle
"""

TIMER = """[Unit]
Description=Hunch periodic indexing

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
"""

DESKTOP = """[Desktop Entry]
Type=Application
Name=Hunch
GenericName=Semantic file search
Comment=Find your files by what they mean
Exec={exe} gui
Icon=system-search
Terminal=false
Categories=Utility;FileTools;
Keywords=search;find;files;semantic;
StartupNotify=true
"""

NAUTILUS_SCRIPT = """#!/usr/bin/env bash
exec {exe} gui
"""


def _xdg(name: str) -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / name


def _exe() -> str:
    return shutil.which("hunch") or "hunch"


def install_user_units() -> list[Path]:
    """systemd --user, so no root and no sudo anywhere.

    No ConditionACPower here: hunch-index.service's single ExecStart runs
    both the cheap catalog pass and the expensive enrichment pass together
    (hunch index --scheduled), so gating the whole unit on AC power would
    block the cheap pass too. AC-power gating instead happens inside
    cmd_index itself (cli.py), via probe.on_ac_power(), so only enrichment
    is deferred on battery.

    Raises RuntimeError if the timer could not actually be enabled -- the
    unit files existing on disk is not the same as indexing ever running,
    and this is the *only* thing that ever triggers enrichment (hunch
    setup never indexes directly), so a silently-swallowed failure here
    means a "successfully installed" system that never indexes anything.
    """
    unit_dir = Path(os.environ.get(
        "XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    service = unit_dir / "hunch-index.service"
    timer = unit_dir / "hunch-index.timer"
    service.write_text(SERVICE.format(exe=_exe()))
    timer.write_text(TIMER)
    if shutil.which("systemctl") is None:
        raise RuntimeError(
            "systemctl not found -- cannot install the background indexing timer")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", "hunch-index.timer"],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "systemctl --user enable --now hunch-index.timer failed: "
            + (result.stderr.strip() or result.stdout.strip() or
               f"exit code {result.returncode}"))
    return [service, timer]


def install_launcher() -> Path:
    apps = _xdg("applications")
    apps.mkdir(parents=True, exist_ok=True)
    dest = apps / "hunch.desktop"
    dest.write_text(DESKTOP.format(exe=_exe()))
    # Cache refresh only, not a functional dependency: the .desktop file
    # itself is already written and will still be picked up (most desktop
    # environments rescan periodically or on next login) even if this is
    # missing or fails -- unlike the systemd timer, this is safe to skip
    # quietly. Unlike check=False, a genuinely missing binary raises
    # FileNotFoundError regardless -- shutil.which avoids that crash.
    if shutil.which("update-desktop-database") is not None:
        subprocess.run(["update-desktop-database", str(apps)], check=False)
    return dest


def install_nautilus_script() -> Path:
    scripts = _xdg("nautilus") / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    dest = scripts / "Search with Hunch"
    dest.write_text(NAUTILUS_SCRIPT.format(exe=_exe()))
    dest.chmod(0o755)
    return dest


def bind_shortcut(binding: str = "<Super>f") -> bool:
    """Bind Super+F via gsettings. Returns False if GNOME is not present."""
    if shutil.which("gsettings") is None:
        return False
    key = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/hunch/"
    base = "org.gnome.settings-daemon.plugins.media-keys"
    try:
        existing = subprocess.run(
            ["gsettings", "get", base, "custom-keybindings"],
            capture_output=True, text=True, check=True).stdout.strip()
        if key not in existing:
            new = "['" + key + "']" if existing in ("@as []", "[]") \
                else existing[:-1] + ", '" + key + "']"
            subprocess.run(["gsettings", "set", base, "custom-keybindings", new],
                           check=True)
        schema = f"{base}.custom-keybinding:{key}"
        for prop, value in (("name", "Hunch"), ("command", f"{_exe()} gui"),
                            ("binding", binding)):
            subprocess.run(["gsettings", "set", schema, prop, value], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
