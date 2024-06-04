from classification_lab.bootstrap import bootstrap_interval


def test_bootstrap_uses_paired_resampling_and_fixed_seed():
    truth = ["a", "b"] * 20
    a = bootstrap_interval(truth, truth, samples=30)
    assert a == bootstrap_interval(truth, truth, samples=30)
    assert a["low"] == a["high"] == a["estimate"] == 1
