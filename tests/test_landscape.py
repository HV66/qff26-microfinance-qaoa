"""Offline tests for scripts.landscape (coarse grids, no network)."""

import numpy as np

from baseline import brute_force
from data import make_portfolio
from qaoa_run import energy_diagonal, fast_probabilities, scaled_cost_diagonal
from qubo import build_qubo
from scripts import landscape

N_BETA, N_GAMMA = 6, 9


def _instance(seed: int = 0):
    pf = make_portfolio(landscape.N_LOANS, seed)
    return pf, build_qubo(pf, landscape.LAM), brute_force(pf, landscape.LAM)


def test_grid_definitions() -> None:
    b, g = landscape.beta_grid(), landscape.gamma_grid()
    assert len(b) == 101 and len(g) == 201
    assert b[0] == 0.0 and b[-1] < np.pi
    assert np.isclose(b[1] - b[0], np.pi / 101)
    assert np.isclose(g[0], -np.pi) and np.isclose(g[-1], np.pi)


def test_probability_never_exceeds_one_and_matches_direct() -> None:
    _, q, bf = _instance()
    p_opt, expect = landscape.surfaces(q, bf.x, N_BETA, N_GAMMA)
    assert p_opt.shape == (N_BETA, N_GAMMA)
    assert p_opt.max() <= 1.0 + 1e-12 and p_opt.min() >= 0.0
    cost_diag, energies = scaled_cost_diagonal(q), energy_diagonal(q)
    betas, gammas = landscape.beta_grid(N_BETA), landscape.gamma_grid(N_GAMMA)
    opt_index = sum(int(v) << k for k, v in enumerate(bf.x))
    low = np.arange(2**q.n_vars) & (2**q.n_loans - 1)
    for i, j in [(0, 0), (2, 4), (5, 8), (3, 1)]:
        probs = fast_probabilities(cost_diag, q.n_vars, np.array([betas[i], gammas[j]]))
        assert np.isclose(p_opt[i, j], probs[low == opt_index].sum())
        assert np.isclose(expect[i, j], probs @ energies)


def test_beta_period_pi_and_param_order() -> None:
    """Verified: beta -> beta + pi leaves the probabilities unchanged, and
    params = [beta, gamma] (beta first). gamma=0 makes beta irrelevant to
    P(optimum) only through the uniform state, so swapping order must differ."""
    _, q, _ = _instance()
    cd = scaled_cost_diagonal(q)
    base = fast_probabilities(cd, q.n_vars, np.array([0.7, 0.4]))
    shifted = fast_probabilities(cd, q.n_vars, np.array([0.7 + np.pi, 0.4]))
    assert np.allclose(base, shifted)
    swapped = fast_probabilities(cd, q.n_vars, np.array([0.4, 0.7]))
    assert not np.allclose(base, swapped)


def test_nearest_index_lookup() -> None:
    n_b, n_g = landscape.N_BETA, landscape.N_GAMMA
    betas, gammas = landscape.beta_grid(), landscape.gamma_grid()
    assert landscape.nearest_index(betas[10], gammas[50]) == (10, 50)
    # beta reduced modulo pi, including negative and > pi values
    assert landscape.nearest_index(betas[10] + np.pi, gammas[50]) == (10, 50)
    assert landscape.nearest_index(betas[10] - np.pi, gammas[50]) == (10, 50)
    # gamma wrapped into [-pi, pi]
    assert landscape.nearest_index(0.0, gammas[20] + 2 * np.pi)[1] == 20
    # beta just below pi is circularly next to index 0
    assert landscape.nearest_index(np.pi - 1e-9, 0.0)[0] == 0
    d = landscape.grid_distance((n_b - 1, 3), (0, 5), n_b)
    assert d == {"beta_steps": 1, "gamma_steps": 2}
    assert n_g == 201


def test_analyse_seed_saved_angle_values_are_exact_not_grid() -> None:
    _, q, bf = _instance()
    params = [3.525339136279876, -0.48673692000357094]
    s, p_opt, expect = landscape.analyse_seed(0, params, N_BETA, N_GAMMA)
    cd, en = scaled_cost_diagonal(q), energy_diagonal(q)
    probs = fast_probabilities(cd, q.n_vars, np.array(params))
    opt_index = sum(int(v) << k for k, v in enumerate(bf.x))
    low = np.arange(2**q.n_vars) & (2**q.n_loans - 1)
    assert np.isclose(
        s["saved_angles"]["p_optimum_exact"], probs[low == opt_index].sum()
    )
    assert np.isclose(s["saved_angles"]["expectation_exact"], probs @ en)
    # beta reduction modulo pi does not change the exact value
    assert np.isclose(
        s["saved_angles"]["p_optimum_exact"],
        s["saved_angles"]["p_optimum_exact_beta_mod_pi"],
    )
    assert np.isclose(s["grid_max_p_optimum"]["value"], p_opt.max())
    assert np.isclose(s["grid_min_expectation"]["value"], expect.min())
    assert s["uniform_p_optimum"] == 1 / 32
    assert s["ratio_saved_p_optimum_to_grid_max"] == (
        s["saved_angles"]["p_optimum_exact"] / p_opt.max()
    )
