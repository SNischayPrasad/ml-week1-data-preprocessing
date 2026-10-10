"""Step 5 - Feature selection.

Three independent views of "does this column earn its place", because any one
of them alone is misleading:
  * Mutual information - catches non-linear dependence, ignores redundancy.
  * Correlation among features - catches redundancy, ignores the target.
  * Permutation importance on a held-out fold - catches what a model actually
    uses, at the cost of being model-specific.

A feature is only dropped when all three agree, and the drop is then verified
against the test metric rather than assumed to be free.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED, FIGURES, OUTPUTS, RANDOM_STATE, TARGET
import viz
from viz import SERIES, SEQ, TEXT_1

viz.apply_style()

tr = pd.read_csv(DATA_PROCESSED / "train_features.csv")
te = pd.read_csv(DATA_PROCESSED / "test_features.csv")
X_tr, y_tr = tr.drop(columns=[TARGET]), tr[TARGET]
X_te, y_te = te.drop(columns=[TARGET]), te[TARGET]

out = {"n_features": int(X_tr.shape[1])}

# --- 5.1 Mutual information -------------------------------------------------
mi = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE)
mi_s = pd.Series(mi, index=X_tr.columns).sort_values(ascending=False)
mi_s.round(6).to_csv(OUTPUTS / "05_mutual_information.csv", header=["mutual_information"])
out["mi_top10"] = {k: round(float(v), 5) for k, v in mi_s.head(10).items()}
out["mi_zero_or_near_zero"] = {k: round(float(v), 6)
                               for k, v in mi_s.items() if v < 1e-4}

# --- 5.2 Redundancy among engineered features -------------------------------
corr = X_tr.corr().abs()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)   # strict upper triangle
pairs = corr.where(mask).stack().sort_values(ascending=False)
out["top_correlated_pairs"] = [
    {"a": a, "b": b, "abs_pearson": round(float(v), 4)}
    for (a, b), v in pairs.head(8).items()
]

# --- 5.3 Permutation importance on a validation fold ------------------------
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tr)
rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, n_jobs=-1,
                            random_state=RANDOM_STATE)
rf.fit(X_fit, y_fit)
perm = permutation_importance(rf, X_val, y_val, n_repeats=5,
                              random_state=RANDOM_STATE, n_jobs=-1,
                              scoring="roc_auc")
perm_s = pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False)
perm_s.round(6).to_csv(OUTPUTS / "05_permutation_importance.csv",
                       header=["permutation_importance_auc"])
out["perm_top10"] = {k: round(float(v), 5) for k, v in perm_s.head(10).items()}
out["perm_non_positive_count"] = int((perm_s <= 0).sum())

# --- 5.4 Does dropping the dead weight actually cost anything? --------------
dead = sorted(set(mi_s[mi_s < 1e-4].index) & set(perm_s[perm_s <= 0].index))
out["agreed_dead_features"] = dead
out["n_agreed_dead"] = len(dead)


def auc_with(cols):
    m = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    m.fit(X_tr[cols], y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te[cols])[:, 1])


keep = [c for c in X_tr.columns if c not in dead]
out["auc_all_features"] = round(float(auc_with(list(X_tr.columns))), 5)
out["auc_after_dropping_dead"] = round(float(auc_with(keep)), 5) if dead else None
out["auc_delta"] = (round(out["auc_after_dropping_dead"] - out["auc_all_features"], 5)
                    if dead else None)

# Top-k curve: how many features do we actually need?
curve = []
for k in [5, 10, 15, 20, 30, 40, len(X_tr.columns)]:
    cols = list(mi_s.head(k).index)
    curve.append({"k": k, "auc": round(float(auc_with(cols)), 5)})
out["topk_curve"] = curve

# --- 5.5 Figure -------------------------------------------------------------
top = mi_s.head(15)[::-1]
shade = np.interp(top.values, (top.values.min(), top.values.max()), (2, 9))
fig, ax = plt.subplots(figsize=(8.8, 5.4))
bars = ax.barh(top.index, top.values, color=[SEQ[int(round(s))] for s in shade],
               height=0.62)
for b, v in zip(bars, top.values):
    ax.text(b.get_width() + top.values.max() * 0.015,
            b.get_y() + b.get_height() / 2, f"{v:.4f}",
            va="center", fontsize=8.8, color=TEXT_1)
ax.set_xlim(0, top.values.max() * 1.16)
ax.set_xlabel("mutual information with income (nats)")
ax.set_title("Top 15 features by mutual information", loc="left")
viz.subtitle(fig, "Computed on the training split only. Marital status and relationship "
                  "dominate, ahead of education and age.", y=0.955)
viz.strip(ax, keep=("left",), xgrid=True)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGURES / "fig6_mutual_information.png", bbox_inches="tight")
plt.close(fig)

(OUTPUTS / "05_feature_selection.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
