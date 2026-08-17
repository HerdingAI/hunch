"""Command line entry point."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import date
from pathlib import Path

from . import budget as budget_mod
from . import catalog, config, db, search as search_mod, worker
from .backends import get_backend
from .setup import install, probe

PRUNE_TOMBSTONE_DAYS = 90
# Comfortably more than budget.RATE_SAMPLE (what the estimate actually
# reads), so trimming never starves the measurement it exists to serve.
TIMING_RETAINED_ROWS = 20000


def _open():
    cfg = config.load_config()
    conn = db.connect(config.db_path(), dim=cfg.embed_dim)
    return conn, cfg


def _human(n: int | None) -> str:
    if not n:
        return "?"
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}P"


def cmd_setup(args) -> int:
    caps = probe.probe()
    verdict = caps.verdict()
    print("Checking your hardware…\n")
    print(f"  GPU     {'yes, %d MB' % caps.vram_mb if caps.has_gpu else 'none detected'}")
    print(f"  Memory  {caps.ram_mb} MB")
    print(f"  Disk    {caps.free_disk_mb} MB free")
    print(f"  CPU     {caps.cpu_count} cores\n")
    print(verdict["summary"] + "\n")

    cfg = config.load_config()
    # Gated on the 4B model's own memory need, not on MIN_VRAM_MB (which
    # answers "is captioning practical", a different question). The 4B
    # weights alone are ~7.5 GB: choosing it for any card over 4 GB meant
    # an 8 GB card OOMed on every file forever, indexing nothing.
    if caps.has_gpu and caps.vram_mb >= probe.EMBED_4B_MIN_VRAM_MB:
        cfg.embed_model = "Qwen/Qwen3-Embedding-4B"
    else:
        # Also the right choice with no GPU: 4B embeddings are
        # impractically slow on CPU.
        cfg.embed_model = "Qwen/Qwen3-Embedding-0.6B"
    config.save_config(cfg)

    # A `pipx install hunch-search` with no extras passes every hardware
    # check above and then silently fails every single file at embed time
    # -- catch it here, loudly, before the timer starts running forever
    # against nothing.
    deps_ok = cfg.backend != "local_inprocess" or probe.local_backend_importable()
    if not deps_ok:
        print(
            "\nWARNING: the local embedding backend's Python packages are not "
            "installed -- every file would fail to index until this is fixed.\n"
            '  pipx install --force "hunch-search[local]"\n'
            "or run `hunch auth openrouter` to use a cloud backend instead.")
    # A narrower variant of the same failure mode: only audio/video files
    # are affected (not every file), but they'd still fail silently and
    # forever without this, exactly like the check above.
    media_ok = cfg.backend == "openrouter" or probe.media_importable()
    if not media_ok:
        print(
            "\nWARNING: faster-whisper is not installed -- audio and video "
            "files would fail to transcribe until this is fixed.\n"
            '  pipx install --force "hunch-search[media]"')

    # Listing a folder that isn't there reads as "this will be indexed".
    # On a machine whose Documents and Downloads live on another drive --
    # a relocated XDG setup, an external disk, a synced folder not yet
    # mounted -- the default list is partly fiction, and the user is told
    # "Nothing else is required" while their actual documents are never
    # touched. Say which ones are real.
    missing = [f for f in cfg.folders if not Path(f).is_dir()]
    print("Folders to index:")
    for folder in cfg.folders:
        print(f"  {folder}" + ("   (not found -- skipped)"
                               if Path(folder) in map(Path, missing) else ""))
    if missing:
        print(f"\n{len(missing)} of these {'folder does' if len(missing) == 1 else 'folders do'} "
              f"not exist and will be skipped. If your files live elsewhere "
              f"(another drive, an external disk), edit `folders` in\n"
              f"  {config.config_path()}\n"
              f"then run `hunch index`.")
    try:
        install.install_user_units()
        timer_ok = True
    except RuntimeError as exc:
        # The timer is the *only* thing that ever triggers enrichment --
        # claiming success here when it silently failed to enable would
        # tell the user everything works while nothing ever indexes.
        timer_ok = False
        print(f"\nWARNING: could not install the background indexing timer: {exc}")
        print("Run `hunch index` manually to index without it.")
    install.install_launcher()
    install.install_nautilus_script()
    bound = install.bind_shortcut()
    print("\nInstalled: " + ("background indexing timer, " if timer_ok else "") +
          "app launcher, Nautilus script")
    print("Shortcut:  " + ("Super+F" if bound else "not set (GNOME not detected)"))
    if timer_ok and deps_ok and media_ok:
        print("\nNothing else is required. Indexing starts automatically.")
    return 0


def _prune(conn) -> dict:
    """Reclaim space from real deletions.

    Every delete or rename produces a tombstoned catalog row, and can strand
    file_embedding/vec_embedding rows with nothing else to clean them up --
    without this the index grows forever even though live search results
    stay correct via the deleted_at join filter in the meantime.

    Order matters: purge tombstones past their retention window FIRST,
    then collect embeddings no catalog row references at all (tombstoned
    or not). A hash still held by a tombstoned-but-not-yet-purged row is
    NOT orphaned -- collecting it anyway breaks two real cases: deleting a
    file, having it pruned, then restoring it byte-identical (a Trash
    restore, a re-added git-tracked file) permanently loses its embedding
    with no re-enrichment path since the row goes straight back to
    status='done' unchanged; and a move/rename, where the old path's row
    is tombstoned in the same crawl that inserts the new path's row --
    if the worker hasn't reached the new row yet when this runs, its
    content_hash is still NULL, so the *only* thing keeping the shared
    embedding alive is the old row's tombstone still referencing it.
    """
    purged = conn.execute(
        "DELETE FROM file_catalog WHERE deleted_at IS NOT NULL "
        "AND deleted_at < unixepoch() - ?",
        (PRUNE_TOMBSTONE_DAYS * 86400,)).rowcount
    conn.execute(
        "DELETE FROM vec_embedding WHERE rowid IN ("
        "  SELECT rowid FROM file_embedding WHERE content_hash NOT IN ("
        "    SELECT content_hash FROM file_catalog "
        "    WHERE content_hash IS NOT NULL))")
    orphaned = conn.execute(
        "DELETE FROM file_embedding WHERE content_hash NOT IN ("
        "  SELECT content_hash FROM file_catalog "
        "  WHERE content_hash IS NOT NULL)").rowcount
    # enrich_timing gains ~2.4 rows per enriched file and nothing ever
    # removed any: 106,322 rows part-way through one real index, and every
    # future re-enrichment adds more, forever. Only recent rows describe the
    # machine as it is now, and that is all seconds_per_file() reads, so keep
    # a window rather than a history.
    conn.execute(
        "DELETE FROM enrich_timing WHERE id < "
        "(SELECT max(id) - ? FROM enrich_timing)", (TIMING_RETAINED_ROWS,))
    conn.commit()
    if orphaned or purged:
        conn.execute("ANALYZE")
    return {"orphaned_vectors": orphaned, "purged_tombstones": purged}


def cmd_index(args) -> int:
    conn, cfg = _open()
    stats = catalog.crawl(conn, cfg)
    print(f"catalog: {stats['seen']:,} seen, {stats['added']:,} added, "
          f"{stats['tombstoned']:,} removed in {stats['seconds']:.1f}s")
    if stats["skipped_roots"]:
        print(f"  skipped {len(stats['skipped_roots'])} unreachable folder(s) "
              f"this run -- their files were left untouched, not removed")
    if args.catalog_only:
        return 0

    if args.scheduled and not probe.on_ac_power():
        # The cheap catalog pass above always runs; only the expensive
        # enrichment pass waits for mains power, so a laptop on battery
        # still gets an up-to-date, searchable-by-name index without
        # burning battery on GPU/CPU-heavy embedding and captioning.
        print("on battery; enrichment deferred until next run on mains power")
        _prune(conn)
        return 0

    today = spent_today = None
    # hunch setup never indexes directly -- it only installs the hourly
    # timer, which always runs with --scheduled -- so the 5-hour first-run
    # budget is unreachable unless a scheduled run can use it too. Tracked
    # as its own cumulative pool (first_pass_spent_seconds), separate from
    # the daily pool, so an hourly timer doesn't grant a fresh 5 hours on
    # every firing: the *whole* first pass is capped at first_run_budget_
    # seconds total, matching "first full index <=5 hours" as a ceiling on
    # the pass, not a per-invocation allowance.
    first_pass_done = args.scheduled and db.get_meta(conn, "first_pass_done") == "1"
    if args.scheduled and not first_pass_done:
        first_pass_spent = float(db.get_meta(conn, "first_pass_spent_seconds") or 0)
        seconds = max(0.0, cfg.first_run_budget_seconds - first_pass_spent)
        if seconds <= 0:
            # The 5-hour ceiling is spent without clearing the backlog --
            # fall through to steady-state daily budgeting rather than
            # spending nothing every hour until someone notices.
            first_pass_done = True
            db.set_meta(conn, "first_pass_done", "1")
    if args.scheduled and first_pass_done:
        # Cumulative same-day spend, not per-invocation: the hourly timer
        # calls this every hour, and each firing budgeting the full daily
        # allowance independently would let actual spend run up to ~24x the
        # promised "20 minutes a day."
        today = date.today().isoformat()
        spent_today = (float(db.get_meta(conn, "budget_spent_today") or 0)
                       if db.get_meta(conn, "budget_day") == today else 0)
        seconds = max(0.0, cfg.daily_budget_seconds - spent_today)
        if seconds <= 0:
            print("daily budget already spent today; nothing more until tomorrow")
            _prune(conn)
            return 0
    if not args.scheduled:
        seconds = cfg.first_run_budget_seconds

    try:
        result = worker.run(conn, cfg, budget_seconds=seconds, limit=args.limit)
    except worker.SystemicFailure as exc:
        # Not a crash: the run stopped itself because every file was failing
        # identically. Say so in one readable sentence rather than a
        # traceback, and leave a non-zero exit so the timer's journal entry
        # is visibly a failure.
        print(f"\nenrichment stopped: {exc}")
        return 1
    print(f"enrichment: {result['processed']:,} files -> {result['counts']}")

    if args.scheduled:
        if not first_pass_done:
            spent = (float(db.get_meta(conn, "first_pass_spent_seconds") or 0)
                     + result["seconds"])
            db.set_meta(conn, "first_pass_spent_seconds", str(spent))
            if budget_mod.next_phase(conn) is None:
                db.set_meta(conn, "first_pass_done", "1")
            elif spent >= cfg.first_run_budget_seconds:
                # The first pass is over and the corpus did not fit in it.
                # From here indexing drops to daily_budget_seconds, which on
                # a large corpus is the difference between "finishes tonight"
                # and "finishes next month" -- measured at 134 files/min on a
                # real 149k-file index, the remaining ~77k files would take
                # about 29 days. Saying nothing leaves the user believing the
                # index is complete when most of it is not.
                left = conn.execute(
                    "SELECT count(*) FROM file_catalog "
                    "WHERE status='pending' AND deleted_at IS NULL").fetchone()[0]
                db.set_meta(conn, "first_pass_done", "1")
                print(
                    f"\nFirst-pass budget "
                    f"({cfg.first_run_budget_seconds // 3600}h) is spent and "
                    f"{left:,} files are still unindexed.\n"
                    f"Indexing continues in the background at "
                    f"{cfg.daily_budget_seconds // 60} minutes a day, so the "
                    f"rest will take a while.\n"
                    f"`hunch status` shows which folders these files are in; "
                    f"narrowing `folders` in\n{config.config_path()} is the "
                    f"fastest way to finish, or run `hunch index` to keep "
                    f"going now.")
        else:
            db.set_meta(conn, "budget_day", today)
            db.set_meta(conn, "budget_spent_today", str(spent_today + result["seconds"]))
        _prune(conn)
    return 0


def cmd_auth(args) -> int:
    if args.provider != "openrouter":
        print(f"unknown provider: {args.provider}")
        return 1
    print(
        "Enabling the OpenRouter backend sends full document text, raw "
        "photo bytes, and raw audio/video bytes to OpenRouter's API for "
        "every file it enriches -- that content leaves this machine. The "
        "local backend never does this.\n")
    try:
        reply = input("Continue and store an API key? [y/N] ")
    except EOFError:
        reply = ""
    if reply.strip().lower() != "y":
        print("cancelled; backend left unchanged")
        return 1
    key = os.environ.get("HUNCH_OPENROUTER_KEY") or getpass.getpass("OpenRouter API key: ")
    if not key.strip():
        print("no key entered; cancelled")
        return 1
    from .backends.openrouter import store_api_key
    store_api_key(key.strip())
    cfg = config.load_config()
    cfg.backend = "openrouter"
    config.save_config(cfg)
    print("key stored and backend set to openrouter")
    return 0


def cmd_search(args) -> int:
    conn, cfg = _open()
    degraded: list[str] = []
    results = search_mod.search(conn, cfg, " ".join(args.query),
                               mode=args.mode, limit=args.limit,
                               degraded=degraded)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return 0 if results else 1
    # Say when only half the search ran. "no matches" from a degraded search
    # is indistinguishable from "your files aren't indexed", and sends the
    # user looking for a problem that isn't there.
    if degraded:
        print(f"note: only filename matching ran -- the meaning-based half "
              f"was unavailable ({degraded[0].splitlines()[0][:120]})",
              file=sys.stderr)
    if not results:
        print("no matches")
        return 1
    for i, r in enumerate(results, 1):
        print(f"{i:2}. [{r.score:.2f}] {r.filename}  ({_human(r.size)})")
        print(f"    {r.path}")
        if r.snippet:
            print(f"    {' '.join(r.snippet.split())[:150]}")
    return 0


def cmd_status(args) -> int:
    conn, cfg = _open()
    rows = dict(conn.execute(
        "SELECT status, count(*) FROM file_catalog WHERE deleted_at IS NULL "
        "GROUP BY status"))
    total = sum(rows.values())
    print(f"{total:,} files catalogued")
    for status in ("done", "pending", "failed", "skipped", "unsupported"):
        print(f"  {status:12} {rows.get(status, 0):,}")

    # Where the index actually came from. On a real machine one folder held
    # 96% of a 149,058-file index -- a scraped-data tree on the Desktop --
    # which spent the first-index budget on machine-generated JSON and
    # buried the user's own documents in every search. Nothing in the tool
    # said so: the counts above look the same whether an index is mostly
    # your documents or mostly one directory you forgot was there.
    if total:
        print("\nWhere they came from")
        for folder in cfg.folders:
            n = conn.execute(
                "SELECT count(*) FROM file_catalog "
                "WHERE deleted_at IS NULL AND path LIKE ?",
                (str(folder).rstrip("/") + "/%",)).fetchone()[0]
            if not n:
                continue
            share = 100 * n / total
            print(f"  {n:>9,}  {share:5.1f}%  {folder}")
            if share >= 50:
                print(f"             ^ most of your index. If that is not "
                      f"what you want searched,\n"
                      f"               remove it from `folders` in "
                      f"{config.config_path()}")

    phase = budget_mod.next_phase(conn)
    print("\nCurrent phase: " +
          (budget_mod.PHASE_LABELS[phase] if phase else "up to date"))

    # "How long will this take" is the question the counts above never
    # answer, and on a large corpus the honest answer is the difference
    # between waiting up and going to bed. The worker has been measuring
    # this all along -- enrich_timing exists precisely to feed planning --
    # so the numbers are real rather than a guess from file counts.
    if phase:
        per_file = budget_mod.seconds_per_file(conn, phase)
        left = budget_mod.pending_count(conn, phase)
        if per_file and left:
            work = left * per_file
            print(f"  {left:,} to go, about {_duration(work)} of processing "
                  f"at the measured {per_file:.2f}s each")
            if db.get_meta(conn, "first_pass_done") == "1":
                # Past the first pass this is not elapsed time: only
                # daily_budget_seconds of it happens per day.
                days = work / max(cfg.daily_budget_seconds, 1)
                if days >= 2:
                    print(f"  at {cfg.daily_budget_seconds // 60} minutes a "
                          f"day that is about {days:.0f} days -- `hunch index` "
                          f"runs it now instead")
    return 0


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def cmd_doctor(args) -> int:
    caps = probe.probe()
    verdict = caps.verdict()
    print("Hardware")
    print(f"  CPU cores      {caps.cpu_count}")
    print(f"  Memory         {caps.ram_mb} MB")
    print(f"  GPU            {caps.vram_mb} MB VRAM" if caps.has_gpu
          else "  GPU            none detected")
    print("\nTools")
    for name, ok in (("pdftotext", caps.has_poppler), ("tesseract", caps.has_tesseract),
                     ("ffmpeg", caps.has_ffmpeg)):
        print(f"  {name:14} {'found' if ok else 'MISSING'}")

    # Distinct from the system-binary checks above: these are the Python
    # extras (`pip install hunch-search[...]`), missing which the affected
    # files fail silently at embed/transcribe time instead of a visible
    # crash -- see probe.local_backend_importable's docstring.
    cfg = config.load_config()
    print("\nPython packages")
    if cfg.backend == "local_inprocess":
        ok = probe.local_backend_importable()
        print(f"  local embedding {'found' if ok else 'MISSING -- pipx install \"hunch-search[local]\"'}")
    if cfg.backend != "openrouter":
        ok = probe.media_importable()
        print(f"  audio/video     {'found' if ok else 'MISSING -- pipx install \"hunch-search[media]\"'}")
    # Not a pip extra: the bindings come from the distro, so a miss here
    # usually means an install without --system-site-packages rather than a
    # missing package. Terminal search still works either way.
    ok = probe.gui_importable()
    print(f"  search window   {'found' if ok else 'MISSING -- sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1'}")

    print("\nWhat works")
    for key in ("documents", "image_text", "photo_descriptions", "transcription"):
        print(f"  {key:20} {'yes' if verdict[key] else 'no'}")
    print("\n" + verdict["summary"])
    return 0


def cmd_reindex(args) -> int:
    conn, cfg = _open()
    if args.embeddings:
        # Re-embed stored text directly, rather than clearing vec_embedding
        # and waiting for the next `hunch index` to rebuild it:
        # enrich_one's dedup fast path treats "file_embedding already has
        # this content_hash" as proof the vector exists too, so a
        # subsequent index run would mark every file "done" again without
        # ever regenerating a vector -- silently leaving semantic search
        # permanently empty. Re-embedding here also avoids stamping
        # embed_model to a value worker.run() can never self-heal from: an
        # empty string is not the same as "no meta row" to
        # embedding_model_matches, so the mismatch guard would reject
        # every future `hunch index` run forever, including a second
        # `hunch reindex --embeddings` attempt.
        backend = get_backend(cfg)
        # A dimension change has no in-place migration: vec0 fixes
        # dimensionality at CREATE and db.connect() uses IF NOT EXISTS, so
        # without this every insert below fails with "Dimension mismatch" --
        # including this command, the one the mismatch message tells people
        # to run. Rebuilding all vectors is exactly what this command does,
        # so recreating the table costs nothing extra.
        current_dim = db.vec_dim(conn)
        if current_dim is not None and current_dim != backend.dim:
            print(f"embedding dimension changed ({current_dim} -> {backend.dim}); "
                  f"rebuilding the vector table")
            db.recreate_vec_table(conn, backend.dim)
        rows = conn.execute(
            "SELECT rowid, content_hash, extracted_text FROM file_embedding").fetchall()
        rebuilt = failed = 0
        for rowid, _chash, text in rows:
            try:
                vector = backend.embed([text or ""])[0]
            except Exception as exc:                       # noqa: BLE001
                # Reindex exists specifically to recover a broken index --
                # a transient failure on one row (e.g. a network hiccup on
                # the openrouter backend) must not discard every vector
                # already rebuilt in this same run.
                failed += 1
                continue
            conn.execute("DELETE FROM vec_embedding WHERE rowid = ?", (rowid,))
            conn.execute("INSERT INTO vec_embedding(rowid, embedding) VALUES (?, ?)",
                         (rowid, db.serialize(vector)))
            conn.commit()
            rebuilt += 1
        if failed:
            # Vectors from different models are not comparable (see
            # db.embedding_model_matches's own docstring) -- stamping
            # embed_model here would silently claim every vector is on the
            # new model when some rows are still on the old one, defeating
            # the mismatch guard entirely. Leave it unset so a rerun (or
            # worker.run()'s own guard) surfaces the incomplete migration
            # instead of hiding it.
            print(f"rebuilt {rebuilt:,} vectors, {failed:,} failed -- "
                  f"run `hunch reindex --embeddings` again to retry")
        else:
            db.set_meta(conn, "embed_model", backend.model_id)
            db.set_meta(conn, "embed_dim", str(backend.dim))
            print(f"rebuilt {rebuilt:,} vectors")
    return 0


def cmd_gui(args) -> int:
    from .gui.app import run_gui
    return run_gui(" ".join(args.query) if args.query else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hunch",
                                     description="Find your files by what they mean.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="probe hardware and install background indexing")

    p_index = sub.add_parser("index", help="catalog and enrich")
    p_index.add_argument("--scheduled", action="store_true",
                         help="use the daily budget instead of the first-run budget")
    p_index.add_argument("--catalog-only", action="store_true")
    p_index.add_argument("--limit", type=int, default=0)

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-m", "--mode", choices=["literal", "semantic", "hybrid"],
                          default="hybrid")
    p_search.add_argument("-n", "--limit", type=int, default=20)
    p_search.add_argument("--json", action="store_true")

    sub.add_parser("status", help="indexing progress")
    sub.add_parser("doctor", help="report hardware, tools and what works")

    p_reindex = sub.add_parser("reindex", help="rebuild vectors from stored text")
    p_reindex.add_argument("--embeddings", action="store_true")

    p_auth = sub.add_parser("auth", help="store credentials for a remote backend")
    p_auth.add_argument("provider", choices=["openrouter"])

    p_gui = sub.add_parser("gui", help="open the search window")
    p_gui.add_argument("query", nargs="*")

    args = parser.parse_args(argv)
    handlers = {"setup": cmd_setup, "index": cmd_index, "search": cmd_search,
                "status": cmd_status, "doctor": cmd_doctor,
                "reindex": cmd_reindex, "auth": cmd_auth, "gui": cmd_gui}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
