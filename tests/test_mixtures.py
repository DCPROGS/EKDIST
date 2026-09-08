"""Tests for ekdist.mixtures.

A mixture fit is one of those functions that cannot look wrong: it returns two
components whatever it is handed, with plausible means and a plausible
proportion, and there is no exception to notice. So it is tested against data
**generated from a known mixture**, where the proportion and both means are
known before the fit runs, rather than against values it produced itself.

Two tests exist to keep the suite honest rather than to check the arithmetic:
:func:`test_a_unimodal_set_is_split_without_a_gap` records that the function
will happily split one population, because that is the mistake a user will
make; and :func:`test_the_recovery_tests_could_fail` shows the tolerances are
tight enough to catch a wrong answer.
"""

import numpy as np
import pytest

from ekdist import mixtures


# --------------------------------------------------------------- fixtures

def two_groups(n=1000, frac=0.27, seed=7):
    """Two log-normal groups in two dimensions, an order of magnitude apart.

    Shaped like Fig. 2 of Colquhoun, Hatton & Hawkes (2003): a majority near
    (2000, 52000) and a minority near (17000, 300000). The majority is
    correlated within itself, as a real one is.
    """
    rng = np.random.default_rng(seed)
    k = int(round(frac * n))
    major = np.exp(rng.multivariate_normal(
        [np.log(2000.0), np.log(52000.0)],
        [[0.08 ** 2, 0.9 * 0.08 * 0.07],
         [0.9 * 0.08 * 0.07, 0.07 ** 2]], n - k))
    minor = np.exp(rng.multivariate_normal(
        [np.log(17000.0), np.log(3.0e5)],
        [[0.10 ** 2, 0.0], [0.0, 0.09 ** 2]], k))
    return np.vstack([major, minor])


# --------------------------------------------------------------- mixture2

def test_recovers_a_known_proportion():
    """27 % was put in; 27 % must come out, within the binomial error."""
    m = mixtures.mixture2(two_groups(n=1000, frac=0.27))
    assert m.converged
    p, se = m.proportion(1)
    assert abs(p - 27.0) < 3 * se, f"{p:.1f} % +- {se:.1f} against 27 %"


@pytest.mark.parametrize("frac", [0.10, 0.27, 0.50])
def test_recovers_the_proportion_at_several_splits(frac):
    """Not tuned to one case: an even split and a small minority both work."""
    m = mixtures.mixture2(two_groups(n=1500, frac=frac, seed=3))
    p, se = m.proportion(1)
    assert abs(p - 100 * frac) < 3 * se


def test_recovers_both_group_means():
    m = mixtures.mixture2(two_groups(n=2000, frac=0.27))
    assert m.means[0][0] == pytest.approx(2000.0, rel=0.05)
    assert m.means[1][0] == pytest.approx(17000.0, rel=0.05)
    assert m.means[0][1] == pytest.approx(52000.0, rel=0.05)
    assert m.means[1][1] == pytest.approx(3.0e5, rel=0.05)


def test_component_zero_is_always_the_lower_one():
    """The ordering is a promise, so that means[0] can be indexed blind."""
    for seed in (11, 17, 23):
        m = mixtures.mixture2(two_groups(seed=seed))
        assert m.means[0][0] < m.means[1][0]
        assert m.weights.sum() == pytest.approx(1.0)


def test_assignment_and_weight_agree_when_the_groups_are_separated():
    m = mixtures.mixture2(two_groups(n=1000, frac=0.27, seed=13))
    p, se = m.proportion(1)
    assert abs(p - 100 * m.weights[1]) < 3 * se


def test_works_on_one_dimension():
    m = mixtures.mixture2(two_groups(n=800)[:, :1])
    p, _ = mixtures.proportion(int((m.labels == 1).sum()), m.labels.size)
    assert 22.0 < p < 32.0


def test_responsibilities_are_a_probability_distribution():
    m = mixtures.mixture2(two_groups(n=400, seed=29))
    assert m.responsibilities.shape == (400, 2)
    np.testing.assert_allclose(m.responsibilities.sum(axis=1), 1.0)
    assert np.all(m.responsibilities >= 0.0)


def test_cluster_selects_by_label_and_takes_any_per_point_quantity():
    X = two_groups(n=500, frac=0.20, seed=31)
    m = mixtures.mixture2(X)
    tag = np.arange(X.shape[0])
    assert m.cluster(tag, 0).size + m.cluster(tag, 1).size == X.shape[0]
    np.testing.assert_array_equal(m.cluster(X[:, 0], 1),
                                  X[m.labels == 1, 0])


def test_is_deterministic():
    """No random restarts, so two runs on the same data agree exactly."""
    X = two_groups(n=600, seed=37)
    a, b = mixtures.mixture2(X), mixtures.mixture2(X)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.means, b.means, rtol=0, atol=0)


def test_rejects_non_positive_values():
    with pytest.raises(ValueError):
        mixtures.mixture2(np.array([[1.0], [2.0], [-1.0], [3.0]]))
    with pytest.raises(ValueError):
        mixtures.mixture2(np.array([[1.0], [2.0], [0.0], [3.0]]))


def test_needs_four_points():
    with pytest.raises(ValueError):
        mixtures.mixture2([1.0, 2.0, 3.0])


def test_a_unimodal_set_is_split_without_a_gap():
    """The mistake a user will make, recorded rather than prevented.

    Nothing here tests whether two components is the right number. One
    population comes back as two whose means are within a factor of two of each
    other -- nothing like the eightfold separation of a real split -- and it is
    that separation, not the absence of an error, that tells the caller.
    """
    rng = np.random.default_rng(5)
    X = np.exp(rng.normal(np.log(2000.0), 0.08, size=(500, 1)))
    m = mixtures.mixture2(X)
    assert m.means[1][0] / m.means[0][0] < 2.0


def test_the_recovery_tests_could_fail():
    """The tolerances above are tight enough to catch a wrong answer.

    If mixture2 returned the overall mean for both components -- the most
    likely way for it to be broken while still looking reasonable -- the
    checks on the group means would reject it.
    """
    X = two_groups(n=2000, frac=0.27)
    overall = float(np.exp(np.log(X[:, 0]).mean()))
    assert not (abs(overall - 2000.0) / 2000.0 < 0.05)
    assert not (abs(overall - 17000.0) / 17000.0 < 0.05)


# -------------------------------------------------------------- gap_split

def test_finds_a_separated_high_tail():
    """234 values below 670, 16 above 37 000: a 55-fold step."""
    body = np.linspace(320.0, 670.0, 234)
    runaway = np.linspace(37000.0, 52000.0, 16)
    x = np.concatenate([body, runaway])

    keep, ratio, cut = mixtures.gap_split(x)
    assert keep.sum() == 234
    assert ratio > 50.0
    assert cut == pytest.approx(670.0)
    assert x[keep].max() < x[~keep].min()


def test_leaves_a_clean_set_alone():
    rng = np.random.default_rng(9)
    x = np.exp(rng.normal(np.log(500.0), 0.2, size=300))
    keep, ratio, cut = mixtures.gap_split(x)
    assert keep.all()
    assert cut is None
    assert ratio < 5.0


def test_the_mask_is_usable_unconditionally():
    """All True when nothing is found, so a caller needs no special case."""
    rng = np.random.default_rng(41)
    x = np.exp(rng.normal(np.log(500.0), 0.2, size=100))
    keep, _, _ = mixtures.gap_split(x)
    assert x[keep].size == x.size


def test_the_threshold_is_honoured():
    x = np.array([1.0, 2.0, 3.0, 12.0])       # a 4-fold gap
    assert mixtures.gap_split(x, min_ratio=5.0)[0].all()
    assert mixtures.gap_split(x, min_ratio=3.0)[0].sum() == 3


def test_the_ratio_is_reported_even_when_no_split_is_taken():
    """So that "no gap" can be stated with a number."""
    x = np.array([1.0, 2.0, 3.0, 12.0])
    keep, ratio, cut = mixtures.gap_split(x, min_ratio=5.0)
    assert keep.all() and cut is None
    assert ratio == pytest.approx(4.0)


def test_the_mask_follows_the_input_order_not_the_sorted_order():
    x = np.array([500.0, 40000.0, 300.0, 45000.0, 400.0])
    keep, _, cut = mixtures.gap_split(x)
    np.testing.assert_array_equal(keep, [True, False, True, False, True])
    assert cut == pytest.approx(500.0)


def test_rejects_non_positive_values():
    with pytest.raises(ValueError):
        mixtures.gap_split([1.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        mixtures.gap_split([1.0, -2.0, 3.0])


# ------------------------------------------------------------- proportion

def test_proportion_and_its_binomial_error():
    p, se = mixtures.proportion(27, 100)
    assert p == pytest.approx(27.0)
    assert se == pytest.approx(100 * np.sqrt(0.27 * 0.73 / 100))


def test_proportion_is_exact_at_the_ends():
    assert mixtures.proportion(0, 50) == (0.0, 0.0)
    assert mixtures.proportion(50, 50) == (100.0, 0.0)
