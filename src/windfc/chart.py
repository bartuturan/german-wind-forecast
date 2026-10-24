"""Fan chart: ensemble forecast quantiles against actual generation."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BAND_COLOR = "#2a6fbb"


def fan_chart(members: pd.DataFrame, actual: pd.Series | None, title: str,
              column: str = "total_mw") -> bytes:
    """PNG bytes. members is the long member table; actual is MW indexed by UTC time."""
    wide = members.pivot_table(index="valid_time", columns="member", values=column) / 1000.0
    q = wide.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    t = wide.index

    fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
    ax.plot(t, wide.to_numpy(), color=BAND_COLOR, alpha=0.06, linewidth=0.7)
    ax.fill_between(t, q[0.1], q[0.9], color=BAND_COLOR, alpha=0.18, linewidth=0, label="P10–P90")
    ax.fill_between(t, q[0.25], q[0.75], color=BAND_COLOR, alpha=0.35, linewidth=0, label="P25–P75")
    ax.plot(t, q[0.5], color=BAND_COLOR, linewidth=2, label="Ensemble median")
    if actual is not None and not actual.dropna().empty:
        a = actual.loc[t.min():t.max()] / 1000.0
        ax.plot(a.index, a.to_numpy(), color="black", linewidth=1.6, label="Actual (Energy-Charts)")
    ax.axvline(t.min(), color="grey", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_ylabel("German wind generation (GW)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_xlabel("Valid time (UTC)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    return _png(fig)


def rank_histogram_chart(pits: dict[str, pd.Series], title: str, bins: int = 10) -> bytes:
    """PNG bytes: one panel per forecast variant, randomized PIT in equal bins against the flat line.

    U shape: under-dispersive. Hump: over-dispersive. Slope: biased (left-heavy = forecast too high).
    """
    fig, axes = plt.subplots(1, len(pits), figsize=(4.2 * len(pits), 3.6), dpi=110,
                             sharey=True, squeeze=False)
    for ax, (label, pit) in zip(axes[0], pits.items()):
        ax.hist(pit, bins=bins, range=(0, 1), color=BAND_COLOR, alpha=0.8, edgecolor="white")
        ax.axhline(len(pit) / bins, color="black", linestyle="--", linewidth=1, label="calibrated")
        ax.set_title(f"{label} (n={len(pit)})")
        ax.set_xlabel("rank of actual among members")
        ax.set_xlim(0, 1)
    axes[0][0].set_ylabel("cases")
    axes[0][0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
