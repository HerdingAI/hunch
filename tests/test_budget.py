import hunch.budget as budget
import hunch.db as db


def test_budget_tracks_and_exhausts():
    b = budget.Budget(10.0)
    assert not b.exhausted()
    b.spend(4.0)
    assert abs(b.remaining() - 6.0) < 1e-6
    b.spend(7.0)
    assert b.exhausted()


def test_phase_order_puts_captioning_last():
    # Captioning is ~26x the cost of the metadata tier, so it must never
    # block cheaper phases from completing.
    assert budget.PHASES[-1] == "image_caption"
    assert budget.PHASES[0] == "document"


def test_next_phase_returns_first_incomplete(tmp_path):
    conn = db.connect(tmp_path / "i.db")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES ('/a.pdf','a.pdf','pdf',100,'pending')")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES ('/b.jpg','b.jpg','jpg',999999,'pending')")
    conn.commit()
    assert budget.next_phase(conn) == "document"

    conn.execute("UPDATE file_catalog SET status='done' WHERE ext='pdf'")
    conn.commit()
    assert budget.next_phase(conn) == "image_meta"


def test_next_phase_returns_none_when_all_done(tmp_path):
    conn = db.connect(tmp_path / "i.db")
    assert budget.next_phase(conn) is None


def test_seconds_per_file_reads_the_timings_nothing_used_to_read(tmp_path):
    # The spec says the worker plans from "its own measured rates" so the
    # budget contract holds "on hardware we have never seen", and names
    # enrich_timing as the table feeding it. Nothing ever read that table:
    # every run wrote ~2.4 rows per file straight into a growing pile. This
    # is the read side. Measured on the live index it gave 0.93 s/file for
    # 101,744 pending documents -- 26 hours, which is the number the user
    # actually needs and nothing else in the tool could tell them.
    conn = db.connect(tmp_path / "i.db", dim=4)
    for _ in range(10):                       # ten files, 0.5s each
        conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds) "
                     "VALUES ('document','hash',0.1)")
        conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds) "
                     "VALUES ('document','extract',0.2)")
        conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds) "
                     "VALUES ('document','embed',0.2)")
    conn.commit()
    assert abs(budget.seconds_per_file(conn, "document") - 0.5) < 1e-6


def test_seconds_per_file_is_none_before_anything_has_been_measured(tmp_path):
    # An estimate invented from no data would be worse than no estimate.
    conn = db.connect(tmp_path / "i.db", dim=4)
    assert budget.seconds_per_file(conn, "document") is None
    assert budget.seconds_per_file(conn, "nonsense") is None
