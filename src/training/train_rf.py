import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from common_features import cls_names, load_features, ms_names, plot_cm


def eval_rf(
    name: str,
    clf: RandomForestClassifier,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    mod_dir: Path,
    fig_dir: Path,
    slug: str,
) -> Tuple[Dict, RandomForestClassifier]:
    t0 = time.time()
    clf.fit(x_tr, y_tr)
    tr_s = time.time() - t0

    yv = clf.predict(x_va)
    v_acc = float(accuracy_score(y_va, yv))
    v_f1 = float(f1_score(y_va, yv, average="macro"))

    t1 = time.time()
    yp = clf.predict(x_te)
    inf_ms = (time.time() - t1) * 1000.0 / max(len(y_te), 1)

    acc = float(accuracy_score(y_te, yp))
    f1m = float(f1_score(y_te, yp, average="macro"))
    f1w = float(f1_score(y_te, yp, average="weighted"))
    cm = confusion_matrix(y_te, yp, labels=[0, 1, 2])
    rep = classification_report(y_te, yp, labels=[0, 1, 2], target_names=cls_names, output_dict=True, zero_division=0)

    plot_cm(cm, f"Random Forest ({name})\nAcc={acc*100:.1f}% | F1={f1m*100:.1f}%", fig_dir / f"matriz_confusion_rf_{slug}.png")
    joblib.dump({"model": clf, "classes": cls_names, "variant": name}, mod_dir / f"rf_{slug}.joblib")

    met = {
        "variant": name,
        "train_time_s": round(tr_s, 3),
        "infer_ms": round(inf_ms, 4),
        "val_accuracy": round(v_acc, 4),
        "val_f1_macro": round(v_f1, 4),
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1m, 4),
        "test_f1_weighted": round(f1w, 4),
        "test_per_class": {
            cn: {
                "precision": round(float(rep[cn]["precision"]), 4),
                "recall": round(float(rep[cn]["recall"]), 4),
                "f1": round(float(rep[cn]["f1-score"]), 4),
                "support": int(rep[cn]["support"]),
            }
            for cn in cls_names
        },
        "confusion_matrix": cm.tolist(),
    }
    return met, clf


def plot_importance(imps: np.ndarray, out_path: Path) -> None:
    order = np.argsort(imps)[-15:]
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=300)
    cols = ["#2a9d8f" if any(k in ms_names[i] for k in ("ndvi", "ndre", "gndvi")) else "#457b9d" for i in order]
    ax.barh([ms_names[i] for i in order], imps[order] * 100.0, color=cols)
    ax.set_xlabel("Importancia relativa (%)", fontsize=11, fontweight="bold")
    ax.set_title("Random Forest - Top 15 Caracteristicas", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for pos, idx in enumerate(order):
        ax.text(imps[idx] * 100.0 + 0.25, pos, f"{imps[idx]*100:.1f}%", va="center", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Entrena y evalua Random Forest.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--pca-hog", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    mod_dir = root / "models" / "random_forest"
    fig_dir = root / "results" / "figures"
    met_dir = root / "results" / "metrics"
    mod_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(root, rebuild=args.rebuild_cache)
    x_hog, x_ms, y, sps = data["x_hog"], data["x_ms"], data["y"], data["splits"]
    m_tr, m_va, m_te = sps == "train", sps == "val", sps == "test"

    def make_rf() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    res = {}
    res["solo_hog"], rf_h = eval_rf("Solo HOG", make_rf(), x_hog[m_tr], y[m_tr], x_hog[m_va], y[m_va], x_hog[m_te], y[m_te], mod_dir, fig_dir, "hog")
    res["multiespectral"], rf_m = eval_rf("Multiespectral + Color", make_rf(), x_ms[m_tr], y[m_tr], x_ms[m_va], y[m_va], x_ms[m_te], y[m_te], mod_dir, fig_dir, "multiespectral")
    plot_importance(rf_m.feature_importances_, fig_dir / "importancia_caracteristicas_rf.png")

    sc_h = StandardScaler().fit(x_hog[m_tr])
    n_c = min(args.pca_hog, int(m_tr.sum()) - 1, x_hog.shape[1])
    pca = PCA(n_components=n_c, random_state=42).fit(sc_h.transform(x_hog[m_tr]))
    xf_tr = np.hstack([pca.transform(sc_h.transform(x_hog[m_tr])), x_ms[m_tr]])
    xf_va = np.hstack([pca.transform(sc_h.transform(x_hog[m_va])), x_ms[m_va]])
    xf_te = np.hstack([pca.transform(sc_h.transform(x_hog[m_te])), x_ms[m_te]])

    res["fusion"], rf_f = eval_rf("Fusion HOG + Multiespectral", make_rf(), xf_tr, y[m_tr], xf_va, y[m_va], xf_te, y[m_te], mod_dir, fig_dir, "fusion")
    joblib.dump({"scaler_hog": sc_h, "pca_hog": pca}, mod_dir / "fusion_transforms_rf.joblib")

    out = {
        "model": "RandomForestClassifier",
        "params": {"n_estimators": args.n_estimators, "max_depth": args.max_depth},
        "split_counts": {"train": int(m_tr.sum()), "val": int(m_va.sum()), "test": int(m_te.sum())},
        "variants": res,
        "ms_feature_importances": {ms_names[i]: round(float(rf_m.feature_importances_[i]), 4) for i in range(len(ms_names))},
    }
    (met_dir / "metricas_test_rf.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key, val in res.items():
        print(f"RF [{key}] Acc={val['test_accuracy']*100:.2f}% | F1={val['test_f1_macro']*100:.2f}%")


if __name__ == "__main__":
    main()
