import numpy as np

from quant_pod.marketmaking.avellaneda_stoikov import AvellanedaStoikovQuoter
from quant_pod.marketmaking.simulator import run_market_maker_sim, summarize


def _quoter(gamma: float) -> AvellanedaStoikovQuoter:
    return AvellanedaStoikovQuoter(gamma=gamma, kappa=1.5, sigma=0.3)


def test_sim_output_lengths_are_consistent():
    result = run_market_maker_sim(_quoter(0.1), mid0=100.0, sigma=0.3, dt=1.0, n_steps=500, seed=7)
    lengths = {len(result.mid), len(result.bid), len(result.ask), len(result.inventory), len(result.cash), len(result.equity)}
    assert lengths == {501}


def test_quotes_straddle_mid_when_flat_at_start():
    result = run_market_maker_sim(_quoter(0.1), mid0=100.0, sigma=0.3, dt=1.0, n_steps=10, seed=7)
    assert result.bid[0] < result.mid[0] < result.ask[0]


def test_equity_matches_cash_plus_marked_inventory():
    result = run_market_maker_sim(_quoter(0.1), mid0=100.0, sigma=0.3, dt=1.0, n_steps=500, seed=7)
    expected = result.cash + result.inventory * result.mid
    assert np.allclose(result.equity, expected)


def test_fills_reconcile_with_inventory_and_cash_paths():
    result = run_market_maker_sim(_quoter(0.1), mid0=100.0, sigma=0.3, dt=1.0, n_steps=2000, seed=7, arrival_rate=50.0)
    assert result.fills  # sanity: this parameterization should produce some fills

    # Rebuild inventory/cash from the fills log alone and check they match the
    # simulator's own running totals -- robust to a bid and ask both filling in
    # the same step, which nets to zero inventory change despite two real fills.
    expected_inventory = np.zeros(len(result.inventory))
    expected_cash = np.full(len(result.cash), result.cash[0])
    for fill in result.fills:
        sign = 1 if fill.side == "buy" else -1
        expected_inventory[fill.step + 1:] += sign * fill.quantity
        expected_cash[fill.step + 1:] -= sign * fill.quantity * fill.price

    assert np.allclose(result.inventory, expected_inventory)
    assert np.allclose(result.cash, expected_cash)


def test_risk_averse_quoter_keeps_tighter_inventory_than_risk_neutral():
    # Same underlying mid path and fill-decision randomness (seed + arrival_rate
    # match) -- only gamma differs -- so any variance gap is attributable to the
    # inventory-skew term, the structural result the AS model is built around.
    risk_averse = run_market_maker_sim(_quoter(2.0), mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=3, arrival_rate=0.5)
    risk_neutral = run_market_maker_sim(_quoter(1e-6), mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=3, arrival_rate=0.5)
    assert np.std(risk_averse.inventory) < np.std(risk_neutral.inventory)


def test_max_inventory_is_never_exceeded_even_with_negligible_skew():
    # gamma ~ 0 means the AS skew alone won't bound inventory -- so if this
    # holds, it's the hard max_inventory overlay doing the work, not the model.
    result = run_market_maker_sim(
        _quoter(1e-6), mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=1,
        arrival_rate=50.0, max_inventory=3.0,
    )
    assert np.max(np.abs(result.inventory)) <= 3.0


def test_quote_is_pulled_on_the_side_that_would_breach_the_limit():
    # Note: arrival_rate=0.3 here (not the 50.0 used elsewhere) -- at dt=1.0 and
    # the ~0.67 half-spread this gamma produces, arrival_rate=50 saturates the
    # fill probability to ~1 on both sides every step, so bid and ask fill
    # together every step and net inventory change is always zero. 0.3 keeps
    # fills probabilistic enough that inventory actually random-walks.
    result = run_market_maker_sim(
        _quoter(1e-6), mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=1,
        arrival_rate=0.3, max_inventory=3.0, fill_size=1.0,
    )
    at_long_limit = result.inventory[:-1] == 3.0
    at_short_limit = result.inventory[:-1] == -3.0
    assert at_long_limit.any() and at_short_limit.any()  # sanity: the limit actually binds both ways
    assert np.isnan(result.bid[:-1][at_long_limit]).all()
    assert np.isnan(result.ask[:-1][at_short_limit]).all()


def test_quote_band_sharply_reduces_requote_rate():
    kwargs = dict(mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=7, arrival_rate=0.5)
    unbanded = run_market_maker_sim(_quoter(0.1), **kwargs)
    banded = run_market_maker_sim(_quoter(0.1), quote_band=0.05, **kwargs)
    assert summarize(unbanded)["bid_requote_rate"] > 0.9  # essentially every step, with no band
    assert summarize(banded)["bid_requote_rate"] < summarize(unbanded)["bid_requote_rate"]


def test_quote_band_none_matches_unbanded_behavior_exactly():
    # quote_band=None (the default) must be a true no-op, not just "a very wide band".
    kwargs = dict(mid0=100.0, sigma=0.3, dt=1.0, n_steps=2000, seed=7, arrival_rate=2.0)
    a = run_market_maker_sim(_quoter(0.1), quote_band=None, **kwargs)
    b = run_market_maker_sim(_quoter(0.1), **kwargs)
    assert np.array_equal(a.bid, b.bid, equal_nan=True)
    assert np.array_equal(a.ask, b.ask, equal_nan=True)


def test_quote_band_still_respects_max_inventory():
    result = run_market_maker_sim(
        _quoter(1e-6), mid0=100.0, sigma=0.3, dt=1.0, n_steps=5000, seed=1,
        arrival_rate=0.3, max_inventory=3.0, quote_band=0.05,
    )
    assert np.max(np.abs(result.inventory)) <= 3.0
