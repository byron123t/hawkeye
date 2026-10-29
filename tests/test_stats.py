"""Checks for hawkeye.stats."""
import numpy as np

from hawkeye.stats import apply_holm, cluster_bootstrap_mean, holm, paired_test


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_paired_test_detects_a_clear_gain():
    rng = np.random.default_rng(1)
    units = np.repeat(np.arange(50), 6)                 # 50 "videos" x 6 frames
    b = rng.uniform(0.3, 0.7, units.size)
    a = b + 0.1 + rng.normal(0, 0.02, units.size)
    r = paired_test(a, b, units, B=2000)
    assert r.n == 300 and r.units == 50
    assert abs(r.diff - 0.1) < 0.01 and r.lo > 0 and r.p_at_floor
    apply_holm([r])
    assert r.verdict == "better"


def test_paired_test_null():
    rng = np.random.default_rng(2)
    units = np.repeat(np.arange(40), 5)
    b = rng.uniform(0, 1, units.size)
    a = b + rng.normal(0, 0.05, units.size)
    r = apply_holm([paired_test(a, b, units, B=2000)])[0]
    assert r.lo < 0 < r.hi and r.verdict == "n.s."


def test_cluster_bootstrap_mean_is_item_weighted():
    ci = cluster_bootstrap_mean([1, 1, 1, 0], ["v1", "v1", "v1", "v2"], B=500)
    assert abs(ci.mean - 0.75) < 1e-12 and ci.units == 2
