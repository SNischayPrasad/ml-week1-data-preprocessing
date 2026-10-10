"""Step 7 - Summary figures: effect sizes, leakage, and final model curves."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, FIGURES, OUTPUTS, RANDOM_STATE, TARGET
import viz
from viz import SERIES, TEXT_1, TEXT_2, TEXT_MUTED, NEUTRAL

viz.apply_style()
E = json.loads((OUTPUTS / "06_experiments.json").read_text())

# Noise floor: standard error of the 5-fold CV mean for the shipped pipeline.
cv_std = E["F_final"]["logreg"]["cv_auc_std"]
se = cv_std / np.sqrt(5)

A, B, C, D = E["A_missing"], E["B_rare_levels"], E["C_scaling"], E["D_log1p"]
effects = [
    ("Scale numerics (KNN)",
     C["knn"]["auc_gain_from_scaling"]),
    ("Drop log1p on capital columns",
     D["without_log1p"]["test_auc"] - D["with_log1p"]["test_auc"]),
    ("'Missing' as a level, not the mode",
     A["explicit_Missing_category"]["test_auc"] - A["most_frequent_imputation"]["test_auc"]),
    ("Scale numerics (logistic reg.)",
     C["logreg"]["auc_gain_from_scaling"]),
    ("Scale numerics (grad. boosting)",
     C["hgb"]["auc_gain_from_scaling"]),
    ("Fold rare levels (vs naive one-hot)",
     B["min_frequency_1pct"]["test_auc"] - B["naive_one_hot"]["test_auc"]),
]
labels = [e[0] for e in effects][::-1]
vals = [e[1] for e in effects][::-1]

fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.axvspan(-se, se, color=NEUTRAL, alpha=0.30, zorder=0)
colors = [SERIES[0] if v >= 0 else SERIES[7] for v in vals]
bars = ax.barh(labels, vals, color=colors, height=0.56, zorder=3)
for b, v in zip(bars, vals):
    off = 0.0007 if v >= 0 else -0.0007
    ax.text(b.get_width() + off, b.get_y() + b.get_height() / 2,
            f"{v:+.5f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=9.2, color=TEXT_1)
ax.axvline(0, color=TEXT_MUTED, lw=1.1, zorder=4)
ax.text(se * 1.08, -0.62, f"+/- 1 standard error of the CV mean ({se:.5f})",
        fontsize=8.8, color=TEXT_2)
ax.set_xlim(-0.004, 0.0235)
ax.set_xlabel("change in test ROC-AUC attributable to the decision")
ax.set_title("What each preprocessing decision was actually worth", loc="left")
viz.subtitle(fig, "Positive means our choice beat the alternative. Anything inside the grey "
                  "band is indistinguishable from fold-to-fold noise.", y=0.955)
viz.strip(ax, keep=("left",), xgrid=True)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIGURES / "fig7_effect_sizes.png", bbox_inches="tight")
plt.close(fig)

# --- Leakage ---------------------------------------------------------------
L = E["E_leakage"]
fig, ax = plt.subplots(figsize=(8.4, 3.3))
items = [("Scaler statistics fitted\non train + test",
          L["E1_scaler_fitted_on_train_plus_test"]["optimism"]),
         ("Target encoding of a 4,233-level key\nfitted before the split",
          L["E2_target_encoding_before_split"]["optimism"])]
bars = ax.barh([i[0] for i in items], [i[1] for i in items],
               color=[NEUTRAL, SERIES[7]], height=0.5)
for b, (_, v) in zip(bars, items):
    ax.text(b.get_width() + 0.0007, b.get_y() + b.get_height() / 2,
            f"{v:+.5f} AUC", va="center", fontsize=9.6, color=TEXT_1)
ax.set_xlim(-0.001, 0.0335)
ax.set_xlabel("false improvement in test AUC caused by the leak")
ax.set_title("Two leaks, two orders of magnitude apart", loc="left")
viz.subtitle(fig, "The same methodological error is nearly harmless for a scaler and "
                  "seriously misleading for a high-cardinality target encoding.", y=0.98)
viz.strip(ax, keep=("left",), xgrid=True)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(FIGURES / "fig8_leakage.png", bbox_inches="tight")
plt.close(fig)

# --- Final model curves ----------------------------------------------------
import sys
sys.path.insert(0, str(OUTPUTS.parent / "src"))
from step06_experiments import make_pre, X_tr, y_tr, X_te, y_te

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
models = [("Logistic regression", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), SERIES[0]),
          ("Gradient boosting", HistGradientBoostingClassifier(random_state=RANDOM_STATE), SERIES[1])]
curves = {}
for name, clf, col in models:
    pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    auc, ap = roc_auc_score(y_te, p), average_precision_score(y_te, p)
    curves[name] = {"auc": round(float(auc), 5), "pr_auc": round(float(ap), 5)}
    axes[0].plot(fpr, tpr, color=col, label=f"{name}  (AUC {auc:.4f})")
    axes[1].plot(rec, prec, color=col, label=f"{name}  (PR-AUC {ap:.4f})")

axes[0].plot([0, 1], [0, 1], color=NEUTRAL, ls=(0, (4, 3)), lw=1.4, label="random (AUC 0.5)")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[0].set_title("ROC curve", loc="left"); axes[0].legend(loc="lower right")
viz.strip(axes[0]); axes[0].grid(True, axis="both")

base = float(y_te.mean())
axes[1].axhline(base, color=NEUTRAL, ls=(0, (4, 3)), lw=1.4,
                label=f"prevalence ({base:.3f})")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Precision-recall curve", loc="left"); axes[1].legend(loc="lower left")
viz.strip(axes[1]); axes[1].grid(True, axis="both")
fig.tight_layout()
fig.savefig(FIGURES / "fig9_final_curves.png", bbox_inches="tight")
plt.close(fig)

(OUTPUTS / "07_summary.json").write_text(json.dumps(
    {"standard_error_cv_mean": round(float(se), 6),
     "effects": {k: round(float(v), 6) for k, v in effects},
     "final_curves": curves}, indent=2))
print(json.dumps({"se": round(float(se), 6), "effects": dict(effects),
                  "final": curves}, indent=2))
print("\nFigures:", [p.name for p in sorted(FIGURES.glob('fig[789]*.png'))])
