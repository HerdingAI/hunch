"""Single-instance libadwaita search window.

Single-instance matters for perceived speed: Super+F activates the running
process instead of relaunching Python and reloading an embedding model, so
search feels instant. The embedder is released after an idle period so the app
does not sit on hundreds of megabytes indefinitely.
"""
from __future__ import annotations

import threading

IDLE_RELEASE_SECONDS = 300
DEBOUNCE_MS = 300

CSS = b"""
.result-name { font-weight: 600; }
.result-path { font-size: 0.8em; opacity: 0.6; }
.result-snip { font-size: 0.8em; font-style: italic; opacity: 0.8; }
"""


def run_gui(initial_query: str = "") -> int:
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Adw, Gdk, GLib, Gtk
    except (ImportError, ValueError) as exc:
        print(f"The Hunch window needs PyGObject and libadwaita: {exc}\n"
              f"Install with:  sudo apt install python3-gi gir1.2-adw-1\n"
              f"Or search from the terminal:  hunch search <query>")
        return 1

    from .. import budget as budget_mod
    from .. import config, db
    from .. import search as search_mod
    from ..backends import get_backend

    class Window(Adw.ApplicationWindow):
        def __init__(self, app, query: str):
            super().__init__(application=app, title="Hunch",
                             default_width=900, default_height=620)
            self._timer = None
            self._seq = 0
            self._cfg = config.load_config()
            self._conn = db.connect(config.db_path(), dim=self._cfg.embed_dim)
            self._backend = None

            view = Adw.ToolbarView()
            header = Adw.HeaderBar()
            view.add_top_bar(header)

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            box.set_margin_top(8), box.set_margin_bottom(8)
            box.set_margin_start(8), box.set_margin_end(8)

            self.entry = Gtk.SearchEntry(
                placeholder_text="Search your files by meaning…")
            self.entry.connect("search-changed", self._on_changed)
            box.append(self.entry)

            self.status = Gtk.Label(xalign=0)
            box.append(self.status)

            self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
            self.list.connect("row-activated", self._activate)
            scroller = Gtk.ScrolledWindow(vexpand=True)
            scroller.set_child(self.list)
            box.append(scroller)

            view.set_content(box)
            self.set_content(view)

            keys = Gtk.EventControllerKey()
            keys.connect("key-pressed", self._on_key)
            self.add_controller(keys)

            if query:
                self.entry.set_text(query)
            GLib.idle_add(self.entry.grab_focus)

        # --- searching ---------------------------------------------------
        def _on_changed(self, *_a):
            if self._timer:
                GLib.source_remove(self._timer)
            self._timer = GLib.timeout_add(DEBOUNCE_MS, self._fire)

        def _fire(self):
            self._timer = None
            text = self.entry.get_text().strip()
            if len(text) < 2:
                # A prompt state, distinct from "searched and found nothing":
                # routing this through _render's empty-results branch would
                # make an unfinished query look identical to a real miss.
                self._clear_results()
                self.status.set_text("" if not text else "keep typing…")
                return False
            self._seq += 1
            seq = self._seq
            self.status.set_text("searching…")
            threading.Thread(target=self._work, args=(text, seq), daemon=True).start()
            return False

        def _work(self, text: str, seq: int):
            try:
                if self._backend is None:
                    self._backend = get_backend(self._cfg)
                results = search_mod.search(self._conn, self._cfg, text,
                                            backend=self._backend)
                err = ""
            except Exception as exc:               # noqa: BLE001
                results, err = [], str(exc)
            # Drop stale responses: a slower earlier query must never overwrite
            # the results of a newer one.
            if seq == self._seq:
                GLib.idle_add(self._render, results, err)

        def _clear_results(self):
            child = self.list.get_first_child()
            while child:
                nxt = child.get_next_sibling()
                self.list.remove(child)
                child = nxt

        def _render(self, results, err):
            self._clear_results()
            if err:
                self.status.set_text(f"error: {err[:120]}")
                return False
            if results:
                self.status.set_text(f"{len(results)} result(s)")
            else:
                # An incomplete first-run index and a genuinely-absent file
                # look identical from an empty result list alone -- tell
                # them apart so the very first thing a new user sees isn't
                # indistinguishable from "Hunch doesn't work."
                phase = budget_mod.next_phase(self._conn)
                if phase:
                    label = budget_mod.PHASE_LABELS[phase].lower()
                    self.status.set_text(
                        f"no matches -- still indexing ({label}); results may be incomplete")
                else:
                    self.status.set_text("no matches")
            for r in results:
                self.list.append(self._row(r))
            first = self.list.get_row_at_index(0)
            if first:
                self.list.select_row(first)
            return False

        def _row(self, r):
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            for margin in ("top", "bottom", "start", "end"):
                getattr(box, f"set_margin_{margin}")(6)
            name = Gtk.Label(label=f"{r.filename}   {r.score:.0%}", xalign=0,
                              ellipsize=3)
            name.add_css_class("result-name")
            box.append(name)
            path = Gtk.Label(label=r.path, xalign=0, ellipsize=3)
            path.add_css_class("result-path")
            box.append(path)
            if r.snippet:
                snip = Gtk.Label(label=" ".join(r.snippet.split())[:160],
                                 xalign=0, ellipsize=3)
                snip.add_css_class("result-snip")
                box.append(snip)
            row.set_child(box)
            row._payload = r
            return row

        # --- actions -----------------------------------------------------
        def _activate(self, _list, row):
            payload = getattr(row, "_payload", None)
            if not payload:
                return
            import subprocess
            from pathlib import Path
            uri = Path(payload.path).as_uri()
            # Nautilus isn't guaranteed on every Ubuntu install (minimal
            # installs, other file managers); `gio open` is the portable
            # fallback so activating a row never just does nothing.
            for cmd in (["nautilus", "--select", uri], ["gio", "open", payload.path]):
                try:
                    subprocess.Popen(cmd, start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except FileNotFoundError:
                    continue
            self.status.set_text("could not open a file manager")

        def _on_key(self, _c, keyval, _code, state):
            if keyval == Gdk.KEY_Escape:
                self.close()
                return True
            if state & Gdk.ModifierType.CONTROL_MASK and keyval == Gdk.KEY_c:
                row = self.list.get_selected_row()
                if row is not None:
                    Gdk.Display.get_default().get_clipboard().set(row._payload.path)
                    self.status.set_text("path copied")
                    return True
            return False

    class App(Adw.Application):
        def __init__(self, query: str):
            super().__init__(application_id="io.github.hunch.Hunch")
            self.query = query
            self.win = None

        def do_activate(self):
            provider = Gtk.CssProvider()
            provider.load_from_data(CSS)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(), provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            if self.win is None:
                self.win = Window(self, self.query)
            self.win.present()

    return App(initial_query).run([])
