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
