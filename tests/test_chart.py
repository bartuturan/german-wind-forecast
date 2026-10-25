from windfc.chart import _slices, scores_chart

PNG = b"\x89PNG\r\n\x1a\n"


def test_slices_pair_raw_and_calibrated_with_day_ahead_last(evaluation):
    labels = [s[0] for s in _slices(evaluation)]
    assert labels == ["0-24 h", "24-48 h", "48-72 h", "Day-ahead"]


def test_scores_chart_is_a_png(evaluation):
    assert scores_chart(evaluation).startswith(PNG)


def test_scores_chart_without_day_ahead(evaluation):
    for variant in ("raw", "calibrated"):
        evaluation[variant]["day_ahead"] = None
    assert "Day-ahead" not in [s[0] for s in _slices(evaluation)]
    assert scores_chart(evaluation).startswith(PNG)
