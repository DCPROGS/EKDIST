"""Statistics of open times and the shut times next to them.

Self-contained: the references here are constructions whose answer is known by
hand or by symmetry, not values taken from another library. The agreement with
the Q-matrix theory is checked separately, where a record can be simulated from
a mechanism.
"""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist import adjacent


# a five-interval record: open shut open shut open
TINTS = np.array([1.0, 10.0, 2.0, 20.0, 3.0])
AMPLS = np.array([5.0, 0.0, 5.0, 0.0, 5.0])


# ------------------------------------------------------------------- pairs

def test_pairs_next():
    """Each opening with the shut time after it; the last has none."""
    topen, tshut = adjacent.pairs(TINTS, AMPLS, side="next")
    npt.assert_array_equal(topen, [1.0, 2.0])
    npt.assert_array_equal(tshut, [10.0, 20.0])


def test_pairs_previous():
    """Each opening with the shut time before it; the first has none."""
    topen, tshut = adjacent.pairs(TINTS, AMPLS, side="previous")
    npt.assert_array_equal(topen, [2.0, 3.0])
    npt.assert_array_equal(tshut, [10.0, 20.0])


def test_pairs_adjacent_counts_the_middle_opening_twice():
    topen, tshut = adjacent.pairs(TINTS, AMPLS, side="adjacent")
    assert len(topen) == 4
    assert sorted(topen.tolist()) == [1.0, 2.0, 2.0, 3.0]


def test_pairs_record_starting_shut():
    tints = np.array([10.0, 1.0, 20.0])
    ampls = np.array([0.0, 5.0, 0.0])
    npt.assert_array_equal(adjacent.pairs(tints, ampls, "next")[1], [20.0])
    npt.assert_array_equal(adjacent.pairs(tints, ampls, "previous")[1], [10.0])


def test_pairs_rejects_nonsense():
    with pytest.raises(ValueError, match="side must be"):
        adjacent.pairs(TINTS, AMPLS, side="sideways")
    with pytest.raises(ValueError, match="same length"):
        adjacent.pairs(TINTS, AMPLS[:-1])


# ------------------------------------------- conditional mean open time

def test_conditional_mean_open_time():
    topen = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    tshut = np.array([0.5, 0.7, 5.0, 6.0, 500.0])
    out = adjacent.conditional_mean_open_time(
        topen, tshut, [(0.0, 1.0), (1.0, 10.0), (10.0, 100.0)])
    npt.assert_array_equal(out["n"], [2, 2, 0])
    npt.assert_allclose(out["open_mean"][:2], [2.0, 15.0])
    npt.assert_allclose(out["shut_mean"][:2], [0.6, 5.5])
    npt.assert_allclose(out["open_sd"][:2],
                        [np.std([1, 3], ddof=1), np.std([10, 20], ddof=1)])
    npt.assert_allclose(out["open_sem"][:2], out["open_sd"][:2] / np.sqrt(2))


def test_empty_range_is_nan_and_keeps_its_place():
    """The result must line up with the ranges, or a plot misaligns."""
    out = adjacent.conditional_mean_open_time(
        np.array([1.0]), np.array([1.0]), [(0.0, 0.5), (0.5, 2.0)])
    assert out["n"][0] == 0
    assert np.isnan(out["open_mean"][0]) and np.isnan(out["shut_mean"][0])
    npt.assert_allclose(out["open_mean"][1], 1.0)


def test_ranges_are_half_open():
    topen = np.array([1.0, 2.0])
    tshut = np.array([1.0, 2.0])
    out = adjacent.conditional_mean_open_time(topen, tshut, [(1.0, 2.0)])
    npt.assert_array_equal(out["n"], [1])
    npt.assert_allclose(out["open_mean"], [1.0])


def test_infinite_upper_limit():
    out = adjacent.conditional_mean_open_time(
        np.array([1.0, 2.0]), np.array([1.0, 1e9]), [(10.0, np.inf)])
    npt.assert_array_equal(out["n"], [1])
    npt.assert_allclose(out["open_mean"], [2.0])


# ------------------------------------------------------------- dependency

def lognormal_pair(n, rho=0.0, seed=0):
    """``n`` pairs of positive times with a given correlation of their logs."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return np.exp(z1 * 1.2 - 6.0), np.exp(z2 * 1.5 - 5.0)


def test_independent_times_give_zero_dependency():
    """The defining property: independence means dependency zero."""
    topen, tshut = lognormal_pair(400000, rho=0.0, seed=1)
    edges_o = np.logspace(np.log10(np.quantile(topen, 0.05)),
                          np.log10(np.quantile(topen, 0.95)), 7)
    edges_s = np.logspace(np.log10(np.quantile(tshut, 0.05)),
                          np.log10(np.quantile(tshut, 0.95)), 7)
    out = adjacent.dependency(topen, tshut, edges_o, edges_s, min_count=50)
    dep = out["dependency"][~np.isnan(out["dependency"])]
    assert len(dep) > 20
    assert abs(dep.mean()) < 0.02, dep.mean()
    assert np.abs(dep).max() < 0.15, np.abs(dep).max()


def test_marginals_are_taken_over_every_pair_not_only_the_grid():
    """Regression: a grid covering part of the distribution must not bias it.

    Taking the marginals as the row and column sums of the joint histogram
    renormalises to whatever rectangle the grid covers. For independent times
    that turned a dependency of zero into values of two or three; a shut-time
    grid never spans the whole distribution, because the gaps between
    activations run to seconds.
    """
    topen, tshut = lognormal_pair(400000, rho=0.0, seed=2)
    # deliberately narrow: most pairs fall outside
    edges_o = np.logspace(np.log10(np.quantile(topen, 0.30)),
                          np.log10(np.quantile(topen, 0.60)), 5)
    edges_s = np.logspace(np.log10(np.quantile(tshut, 0.30)),
                          np.log10(np.quantile(tshut, 0.60)), 5)
    out = adjacent.dependency(topen, tshut, edges_o, edges_s, min_count=50)
    assert out["inside"] < 0.2 * out["n"], "the grid should be narrow"
    dep = out["dependency"][~np.isnan(out["dependency"])]
    assert np.abs(dep).max() < 0.15, np.abs(dep).max()


def test_correlated_times_give_positive_dependency_on_the_diagonal():
    """Both short or both long together, one of each much less than chance."""
    topen, tshut = lognormal_pair(400000, rho=0.6, seed=3)
    edges_o = np.logspace(np.log10(np.quantile(topen, 0.05)),
                          np.log10(np.quantile(topen, 0.95)), 7)
    edges_s = np.logspace(np.log10(np.quantile(tshut, 0.05)),
                          np.log10(np.quantile(tshut, 0.95)), 7)
    dep = adjacent.dependency(topen, tshut, edges_o, edges_s,
                              min_count=20)["dependency"]
    assert dep[0, 0] > 0.5 and dep[-1, -1] > 0.5
    # the opposite corners are strongly negative, or so nearly empty that they
    # come back nan -- which says the same thing
    for corner in (dep[0, -1], dep[-1, 0]):
        assert np.isnan(corner) or corner < -0.5, corner
    # and the effect is monotone across the grid
    assert np.nanmean(np.diag(dep)) > np.nanmean(np.diag(np.fliplr(dep)))


def test_sparse_cells_are_nan():
    topen, tshut = lognormal_pair(2000, seed=4)
    edges_o = np.logspace(np.log10(topen.min()), np.log10(topen.max()), 30)
    edges_s = np.logspace(np.log10(tshut.min()), np.log10(tshut.max()), 30)
    out = adjacent.dependency(topen, tshut, edges_o, edges_s, min_count=10)
    dep, counts = out["dependency"], out["counts"]
    assert np.isnan(dep[counts < 10]).all()
    assert not np.isnan(dep[counts >= 10]).any()


def test_empty_grid_raises():
    with pytest.raises(ValueError, match="no pairs"):
        adjacent.dependency(np.array([1.0]), np.array([1.0]),
                            np.array([10.0, 20.0]), np.array([10.0, 20.0]))


def test_log_edges():
    values = np.array([1e-5, 1e-3, 1e-1])
    edges = adjacent.log_edges(values, tres=1e-5, per_decade=2)
    assert edges[0] == pytest.approx(1e-5)
    assert edges[-1] == pytest.approx(1e-1)
    assert len(edges) == 9
    assert np.all(np.diff(edges) > 0)
