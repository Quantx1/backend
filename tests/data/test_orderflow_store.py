from backend.data.orderflow_store import upsert_rows


class FakeTable:
    def __init__(self): self.calls = []
    def upsert(self, rows, on_conflict=None): self.calls.append((rows, on_conflict)); return self
    def execute(self): return self


class FakeSB:
    def __init__(self): self.t = FakeTable(); self.last = None
    def table(self, name): self.last = name; return self.t


def test_upsert_rows_calls_supabase():
    sb = FakeSB()
    n = upsert_rows(sb, "participant_oi_eod", [{"date": "d", "participant": "fii"}], "date,participant")
    assert n == 1 and sb.last == "participant_oi_eod"
    assert sb.t.calls[0][1] == "date,participant"


def test_upsert_rows_empty_noop():
    sb = FakeSB()
    assert upsert_rows(sb, "x", [], "k") == 0
