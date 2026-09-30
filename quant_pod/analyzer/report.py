"""Turns a metrics summary + equity curve into a printable report and a saved plot."""
from __future__ import annotations

import re

import pandas as pd

from quant_pod.analyzer import metrics as m
from quant_pod.common.config import REPORTS_DIR


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def format_summary(name: str, summary: dict) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"Total return:  {summary['total_return']:>8.2%}")
    lines.append(f"CAGR:          {summary['cagr']:>8.2%}")
    lines.append(f"Sharpe:        {summary['sharpe']:>8.2f}")
    lines.append(f"Sortino:       {summary['sortino']:>8.2f}")
    lines.append(f"Max drawdown:  {summary['max_drawdown']:>8.2%}")
    lines.append(f"Calmar:        {summary['calmar']:>8.2f}")
    lines.append(f"VaR (95%):     {summary['var_95']:>8.2%}")
    lines.append(f"Turnover:      {summary['turnover']:>8.2f}x")
    lines.append(f"Num trades:    {summary['num_trades']:>8d}")
    return "\n".join(lines)


def save_equity_plot(name: str, equity_curve: pd.Series) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve.plot(ax=ax, title=f"{name} — Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    fig.tight_layout()

    out_path = REPORTS_DIR / f"{_slug(name)}_equity.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def generate_report(name: str, equity_curve: pd.Series, returns: pd.Series, trades: pd.DataFrame) -> dict:
    summary = m.summarize(equity_curve, returns, trades)
    text = format_summary(name, summary)
    plot_path = save_equity_plot(name, equity_curve)
    print(text)
    print(f"Equity plot saved to {plot_path}")
    return {"summary": summary, "text": text, "plot_path": plot_path}
