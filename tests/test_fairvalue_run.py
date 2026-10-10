import json

from fairvalue.run import run


def test_run_offline(tmp_path):
    h = run(tmp_path, offline=True)           # fixtures: ES/GC/CL exist, spot proxies may not
    assert (tmp_path / "headline.json").exists()
    assert json.loads((tmp_path / "headline.json").read_text()) == h
    assert (tmp_path / "fairvalue.xlsx").exists()
    assert h["ES_days"] > 100 and h["ES_mean_abs_gap_bps"] < 500
