"""Step 3 - Exploratory analysis. Each figure exists to justify one decision."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_INTERIM, DATA_PROCESSED, FIGURES, OUTPUTS, TARGET
import viz
from viz import SERIES, SEQ, TEXT_1, TEXT_MUTED, NEUTRAL, SURFACE

viz.apply_style()
eda = {}

raw = pd.read_csv(DATA_INTERIM / "adult_raw_combined.csv", skipinitialspace=True)
tr = pd.read_csv(DATA_PROCESSED / "train_clean.csv")

# ---------------------------------------------------------------- Figure 1
# Missingness is concentrated, nested, and NOT random.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
cols = ["occupation", "workclass", "native_country"]
pct = [100 * raw[c].eq("?").mean() for c in cols]
cnt = [int(raw[c].eq("?").sum()) for c in cols]
ax = axes[0]
bars = ax.barh(cols, pct, color=SERIES[0], height=0.55)
for b, p, n in zip(bars, pct, cnt):
    ax.text(b.get_width() + 0.12, b.get_y() + b.get_height() / 2,
            f"{p:.2f}%  (n={n:,})", va="center", fontsize=9.5, color=TEXT_1)
ax.set_xlim(0, max(pct) * 1.45)
ax.set_title("Missing values hide behind a '?' string", loc="left")
ax.set_xlabel("share of rows")
viz.strip(ax, keep=("left",), xgrid=True)
ax.invert_yaxis()

mask = raw[cols].eq("?").any(axis=1)
grp = ["Rows with no\nmissing values", "Rows with >=1\nmissing value"]
rate = [100 * (raw.loc[~mask, TARGET] == ">50K").mean(),
        100 * (raw.loc[mask, TARGET] == ">50K").mean()]
nrows = [int((~mask).sum()), int(mask.sum())]
ax = axes[1]
bars = ax.bar(grp, rate, color=[SERIES[0], SERIES[1]], width=0.5)
for b, r, n in zip(bars, rate, nrows):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{r:.1f}%",
            ha="center", fontsize=11.5, fontweight="bold", color=TEXT_1)
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 2.0, f"n={n:,}",
            ha="center", fontsize=9, color=SURFACE)
ax.set_ylim(0, 30)
ax.set_ylabel("share earning >50K")
ax.set_title("Missingness is not random", loc="left")
viz.strip(ax)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_missingness.png", bbox_inches="tight")
plt.close(fig)
eda["fig1_rate_complete"] = round(rate[0], 2)
eda["fig1_rate_incomplete"] = round(rate[1], 2)

# ---------------------------------------------------------------- Figure 2
# The IQR rule is wrong for hours_per_week: Q1 == Q3 == 40.
h = raw["hours_per_week"]
q1, q3 = h.quantile(.25), h.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
flagged = (h < lo) | (h > hi)
fig, ax = plt.subplots(figsize=(9.2, 4.3))
bins = np.arange(0, 102, 2)
ax.hist([h[~flagged], h[flagged]], bins=bins, stacked=True,
        color=[SERIES[0], SERIES[7]], edgecolor=SURFACE, linewidth=0.6,
        label=["kept by the 1.5xIQR rule",
               f"condemned as an outlier ({100 * flagged.mean():.1f}% of rows)"])
ax.annotate(f"Q1 = 40 h, Q3 = 45 h, so IQR = {iqr:.0f}\nfence = [{lo:.1f} h, {hi:.1f} h]",
            xy=(60.5, 2400), xytext=(66, 11000), fontsize=9.6, color=TEXT_1,
            arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=1.2))
ax.set_xlabel("hours worked per week")
ax.set_ylabel("number of people")
ax.set_title("Why we did not auto-remove outliers", loc="left")
viz.subtitle(fig, "46.7% of the sample works exactly 40 h, so Q1 and the median sit on the "
                  "same value, the IQR shrinks to 5 h, and the fence condemns 13,496 "
                  "ordinary people.", y=0.935)
ax.legend(loc="upper right")
viz.strip(ax)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIGURES / "fig2_iqr_trap.png", bbox_inches="tight")
plt.close(fig)
eda["fig2_hours_flagged_pct"] = round(100 * float(flagged.mean()), 2)
eda["fig2_hours_flagged_n"] = int(flagged.sum())
eda["fig2_pct_exactly_40"] = round(100 * float((h == 40).mean()), 2)

# ---------------------------------------------------------------- Figure 3
# capital_gain: zero-inflated and top-coded at 99999.
cg = raw["capital_gain"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                         gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
shares = [100 * (cg == 0).mean(),
          100 * ((cg > 0) & (cg < 99999)).mean(),
          100 * (cg == 99999).mean()]
counts = [int((cg == 0).sum()), int(((cg > 0) & (cg < 99999)).sum()),
          int((cg == 99999).sum())]
bars = ax.bar(["exactly 0", "1 - 41,310", "99,999"], shares,
              color=[NEUTRAL, SERIES[0], SERIES[7]], width=0.55)
for b, p, n in zip(bars, shares, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8,
            f"{p:.2f}%\nn={n:,}", ha="center", fontsize=9.3, color=TEXT_1)
ax.set_ylim(0, 110)
ax.set_ylabel("share of rows")
ax.set_title("capital_gain is 91.7% zeros", loc="left")
viz.strip(ax)

ax = axes[1]
nz = cg[(cg > 0) & (cg < 99999)]
ax.hist(nz, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
ax.axvline(41310, color=TEXT_MUTED, ls=(0, (4, 3)), lw=1.4)
ax.annotate("largest genuine value: 41,310", xy=(41310, 250), xytext=(21000, 540),
            fontsize=9.2, color=TEXT_1,
            arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=1.1))
ax.text(0.03, 0.80, "244 rows sit at 99,999 - a censoring code,\n"
                    "not an amount. 100% of them earn >50K.",
        transform=ax.transAxes, fontsize=9.2, color=SERIES[7])
ax.set_xlabel("capital gain (non-zero, censoring code excluded)")
ax.set_ylabel("number of people")
ax.set_title("A 58,689 gap separates 99,999 from real data", loc="left")
viz.strip(ax)
fig.tight_layout()
fig.savefig(FIGURES / "fig3_capital_gain.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# Scale disparity -> why standardisation is required for distance/gradient models.
num = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
stds = tr[num].std().sort_values()
fig, ax = plt.subplots(figsize=(8.6, 3.6))
bars = ax.barh(stds.index, stds.values, color=SERIES[0], height=0.55)
for b, v in zip(bars, stds.values):
    ax.text(b.get_width() * 1.18, b.get_y() + b.get_height() / 2, f"{v:,.2f}",
            va="center", fontsize=9.4, color=TEXT_1)
ax.set_xscale("log")
ax.set_xlim(1, stds.max() * 9)
ax.set_xlabel("standard deviation (log scale)")
ax.set_title(f"Raw features differ in spread by {stds.max() / stds.min():,.0f}x", loc="left")
viz.subtitle(fig, "Left untouched, capital_gain dominates every distance- and "
                  "gradient-based model.", y=0.90)
viz.strip(ax, keep=("left",), xgrid=True)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIGURES / "fig4_scale.png", bbox_inches="tight")
plt.close(fig)
eda["fig4_std_ratio"] = round(float(stds.max() / stds.min()), 1)

# ---------------------------------------------------------------- Figure 5
# Education is ordinal - one-hot would throw the ordering away.
g = tr.groupby("education_num")[TARGET].agg(["mean", "size"]).reset_index()
shade = np.interp(g["mean"], (g["mean"].min(), g["mean"].max()), (1, 9))
colors = [SEQ[int(round(s))] for s in shade]
fig, ax = plt.subplots(figsize=(9.2, 4.0))
ax.bar(g["education_num"], 100 * g["mean"], color=colors, width=0.68)
for x, m, n in zip(g["education_num"], g["mean"], g["size"]):
    if n > 400:
        ax.text(x, 100 * m + 1.6, f"{100 * m:.0f}%", ha="center", fontsize=8.6,
                color=TEXT_1)
ax.set_xticks(g["education_num"])
ax.set_xlabel("education_num   (1 = Preschool  ->  16 = Doctorate)")
ax.set_ylabel("share earning >50K")
ax.set_ylim(0, 85)
ax.set_title("education_num is near-monotonic, so we keep it as a number", loc="left")
viz.subtitle(fig, "Spearman rho = 0.968 between level and >50K rate (4 minor inversions, "
                  "largest at 15->16). One-hot would discard this ordering.", y=0.925)
viz.strip(ax)
fig.tight_layout(rect=[0, 0, 1, 0.89])
fig.savefig(FIGURES / "fig5_education_ordinal.png", bbox_inches="tight")
plt.close(fig)
eda["fig5_rate_at_1"] = round(100 * float(g.loc[g.education_num == 1, "mean"].iloc[0]), 2)
eda["fig5_rate_at_16"] = round(100 * float(g.loc[g.education_num == 16, "mean"].iloc[0]), 2)

OUTPUTS.joinpath("03_eda.json").write_text(json.dumps(eda, indent=2))
print(json.dumps(eda, indent=2))
print("\nFigures written:")
for p in sorted(FIGURES.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
