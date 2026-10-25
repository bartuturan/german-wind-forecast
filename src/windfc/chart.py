"""Charts: the fan chart (ensemble quantiles against actuals), the rank histogram and the
verification scorecard."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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


RAW_COLOR = "#8f8e88"    # raw ensemble: de-emphasised grey, hollow markers
INK, QUIET = "#0b0b0b", "#52514e"
GRID, REF = "#e6e5e1", "#a3a29c"

# (score key, panel title, how to read it, value scale, reference line, value format)
SCORE_PANELS = [
    ("mae_mw", "MAE of the median (GW)", "lower is better", 1e-3, None, "{:.1f}"),
    ("rmse_mw", "RMSE of the median (GW)", "lower is better", 1e-3, None, "{:.1f}"),
    ("bias_mw", "Bias of the median (GW)", "closer to 0 is better", 1e-3, (0.0, "no bias"), "{:+.1f}"),
    ("crps_mw", "CRPS (GW)", "lower is better", 1e-3, None, "{:.1f}"),
    ("coverage_p10_p90", "P10–P90 coverage (%)", "closer to 80% is better", 100, (80.0, "target"), "{:.0f}%"),
    ("crpss_vs_persistence", "CRPS skill vs persistence", "higher is better, 1 = perfect", 1,
     (0.0, "= persistence"), "{:.2f}"),
]


def _slices(evaluation: dict) -> list[tuple[str, dict, dict]]:
    """(label, raw scores, calibrated scores) per lead-time bin, then the day-ahead window."""
    raw, cal = evaluation["raw"], evaluation["calibrated"]
    cal_by_lead = {r["lead_h"]: r for r in cal["by_lead"]}
    out = [(f"{r['lead_h']} h", r, cal_by_lead[r["lead_h"]]) for r in raw["by_lead"]
           if r["lead_h"] in cal_by_lead]
    if raw.get("day_ahead") and cal.get("day_ahead"):
        out.append(("Day-ahead", raw["day_ahead"], cal["day_ahead"]))
    return out


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REF)
    ax.tick_params(colors=QUIET, labelsize=9, length=0)


def scores_chart(evaluation: dict) -> bytes:
    """PNG bytes: the pooled verification scores of scripts/calibrate.py (data/evaluation.json).

    Six panels, one score each, raw against calibrated per lead-time bin and for the day-ahead
    window; below them, the bias of every scored run, which shows how the bias drifts.
    """
    slices = _slices(evaluation)
    labels = [s[0] for s in slices]
    xs = range(len(slices))
    da = labels.index("Day-ahead") if "Day-ahead" in labels else None

    fig = plt.figure(figsize=(13, 10), dpi=110)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.28,
                            left=0.06, right=0.97, top=0.86, bottom=0.07)
    for i, (key, title, how, scale, ref, fmt) in enumerate(SCORE_PANELS):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        _style(ax)
        raw = [s[1][key] * scale if s[1][key] is not None else float("nan") for s in slices]
        cal = [s[2][key] * scale if s[2][key] is not None else float("nan") for s in slices]
        if da is not None:
            ax.axvspan(da - 0.5, da + 0.5, color="#f2f1ee", zorder=0)
        if ref is not None:
            ax.axhline(ref[0], color=REF, linewidth=1, zorder=1)
            ax.annotate(ref[1], (0, ref[0]), xycoords=("axes fraction", "data"), xytext=(3, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8, color=QUIET)
        ax.vlines(xs, raw, cal, color=GRID, linewidth=2, zorder=2)
        ax.plot(xs, raw, "o", markersize=8, markerfacecolor="white", markeredgecolor=RAW_COLOR,
                markeredgewidth=1.8, linestyle="none", zorder=3)
        ax.plot(xs, cal, "o", markersize=9, color=BAND_COLOR, markeredgecolor="white",
                markeredgewidth=1.2, linestyle="none", zorder=4)
        if da is not None:  # label only the window a trader acts on
            for value, color, weight in ((raw[da], QUIET, "normal"), (cal[da], INK, "bold")):
                ax.annotate(fmt.format(value), (da, value), xytext=(9, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=color, fontweight=weight)
        lo = min(min(raw), min(cal), ref[0] if ref else 0, 0)
        hi = max(max(raw), max(cal), ref[0] if ref else 0)
        pad = 0.12 * (hi - lo or 1)
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        if key == "coverage_p10_p90":
            ax.set_ylim(0, 100)
        ax.set_xticks(list(xs), labels)
        ax.set_xlim(-0.6, len(slices) - 0.1)
        ax.set_title(f"{title}\n", loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(0, 1.04, how, transform=ax.transAxes, fontsize=8.5, color=QUIET)

    # bias per run: the drift the expanding-window calibration has to chase
    ax = fig.add_subplot(grid[2, :])
    _style(ax)
    runs = pd.to_datetime([r["run"] for r in evaluation["raw"]["by_run"]], format="%Y-%m-%dT%H")
    for variant, color, face in (("raw", RAW_COLOR, "white"), ("calibrated", BAND_COLOR, BAND_COLOR)):
        bias = [r["bias_mw"] / 1000.0 for r in evaluation[variant]["by_run"]]
        ax.plot(runs, bias, color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.5, zorder=3)
    ax.axhline(0, color=REF, linewidth=1, zorder=1)
    ax.annotate("no bias", (0, 0), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=QUIET)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("Run base time (UTC)", fontsize=9, color=QUIET)
    ax.set_title("Bias of the median per run (GW)\n", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK)
    ax.text(0, 1.04, "each point is one run, all its lead times; closer to 0 is better",
            transform=ax.transAxes, fontsize=8.5, color=QUIET)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="white",
               markeredgecolor=RAW_COLOR, markeredgewidth=1.8, label="Raw ensemble"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=BAND_COLOR,
               label="Calibrated (scales fitted on earlier runs only)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.935), ncol=2,
               frameon=False, fontsize=9.5)
    scored = evaluation["runs_scored"]
    fig.suptitle(f"Verification scores, German wind ({evaluation['fleet']}), {len(scored)} runs scored "
                 f"({scored[0]} to {scored[-1]})", x=0.06, y=0.975, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    return _png(fig)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
