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
            self._idle_timer = None
            self._seq = 0
            self._cfg = config.load_config()
            # Main-thread-only: _render (dispatched via GLib.idle_add, so
            # it always runs on the main loop) is the only place this
            # connection is touched.
            self._conn = db.connect(config.db_path(), dim=self._cfg.embed_dim)
            # _work runs in a spawned thread, so it needs its own connection
            # (opened with check_same_thread=False) rather than sharing
            # self._conn. A *dedicated, reused* one, not a fresh open+close
            # per search: SQLite's unix VFS keeps a per-process registry of
            # every open file descriptor on a given inode so it can protect
            # POSIX advisory locks (closing any one fd on a file silently
            # drops every lock the whole process holds on it, POSIX-wide --
            # not just that fd's). As long as self._conn above stays open
            # for the app's lifetime, that protection means closing a
            # throwaway per-search connection never actually released its
            # own file descriptor either -- confirmed live: 8 open-then-
            # closed search connections left 8 permanently open fds on
            # index.db, all reclaimed at once only when the last connection
            # to the file (self._conn) finally closed. A long GUI session
            # doing many searches would slowly exhaust the process's file
            # descriptor limit. One connection, opened once and reused
            # under a lock, opens exactly one extra fd for the app's whole
            # life instead of one per search.
            self._search_conn = db.connect(config.db_path(), dim=self._cfg.embed_dim,
                                           check_same_thread=False)
            self._search_lock = threading.Lock()
            self._backend = None
            self._backend_lock = threading.Lock()
            # Closing the window (Escape) hides it instead of destroying
            # it; app.hold() (in App.do_activate) keeps the process alive
            # regardless, so a later `hunch gui` re-activates this same
            # process via GApplication's single-instance handling instead
            # of relaunching Python and reloading the embedding model.
            self.set_hide_on_close(True)

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
            # grab_focus() returns whether focus succeeded, not "call me
            # again" -- GLib idle callbacks repeat while their return value
            # is truthy, so returning it directly turns this into a busy
            # loop consuming a full CPU core for as long as the window
            # stays open (reproduced: ~265k calls/sec on a real display).
            GLib.idle_add(lambda: self.entry.grab_focus() and False)

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
            self._reset_idle_timer()
            threading.Thread(target=self._work, args=(text, seq), daemon=True).start()
            return False

        def _get_backend(self):
            # Double-checked locking: the common case (already loaded)
            # never blocks on the lock, and two searches firing close
            # together can't both pay for loading the embedding model or
            # race on which one's backend instance wins.
            if self._backend is None:
                with self._backend_lock:
                    if self._backend is None:
                        self._backend = get_backend(self._cfg)
            return self._backend

        def _reset_idle_timer(self):
            if self._idle_timer:
                GLib.source_remove(self._idle_timer)
            self._idle_timer = GLib.timeout_add_seconds(
                IDLE_RELEASE_SECONDS, self._release_backend)

        def _release_backend(self):
            self._idle_timer = None
            with self._backend_lock:
                if self._backend is not None:
                    self._backend.release()
                    self._backend = None
            return False

        def _work(self, text: str, seq: int):
            try:
                backend = self._get_backend()
                # self._search_conn is shared by every search thread, so
                # only one search actually touches SQLite at a time -- WAL
                # mode still lets it run concurrently with the main
                # thread's self._conn and with a separate background
                # indexer process, just not with another _work() thread.
                with self._search_lock:
                    results = search_mod.search(self._search_conn, self._cfg,
                                                text, backend=backend)
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
            if self.win is None:
                provider = Gtk.CssProvider()
                provider.load_from_data(CSS)
                Gtk.StyleContext.add_provider_for_display(
                    Gdk.Display.get_default(), provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
                # Keeps the process alive after the window is hidden
                # (Escape / set_hide_on_close above) -- without this,
                # GApplication exits once its last window closes, and the
                # whole point of single-instance activation (skip the
                # model reload) is lost on every reopen.
                self.hold()
                self.win = Window(self, self.query)
            self.win.present()

    return App(initial_query).run([])
