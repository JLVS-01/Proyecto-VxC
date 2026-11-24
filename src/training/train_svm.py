import argparse
import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common_features import cls_names, load_features, plot_cm


def tune_svm(
    name: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    mod_dir: Path,
    fig_dir: Path,
    slug: str,
    pca_n: int = 0,
) -> Dict:
    best_cfg = None
    best_f1 = -1.0
    best_acc = 0.0
    best_pipe = None

    t0 = time.time()
    for c_val in [0.5, 1.0, 3.0, 10.0, 30.0]:
        for g_val in ["scale", 0.01, 0.003]:
            steps = [("scaler", StandardScaler())]
            if pca_n > 0:
                n_c = min(pca_n, len(y_tr) - 1, x_tr.shape[1])
                steps.append(("pca", PCA(n_components=n_c, random_state=42)))
            steps.append(("svm", SVC(kernel="rbf", C=c_val, gamma=g_val, class_weight="balanced", random_state=42)))
            pipe = Pipeline(steps)
            pipe.fit(x_tr, y_tr)
            yv = pipe.predict(x_va)
            vf1 = float(f1_score(y_va, yv, average="macro"))
            if vf1 > best_f1:
                best_f1 = vf1
                best_acc = float(accuracy_score(y_va, yv))
                best_cfg = {"c": c_val, "gamma": g_val, "pca": pca_n}
                best_pipe = pipe

    tune_s = time.time() - t0
    t1 = time.time()
    yp = best_pipe.predict(x_te)
    inf_ms = (time.time() - t1) * 1000.0 / max(len(y_te), 1)

    acc = float(accuracy_score(y_te, yp))
    f1m = float(f1_score(y_te, yp, average="macro"))
    f1w = float(f1_score(y_te, yp, average="weighted"))
    cm = confusion_matrix(y_te, yp, labels=[0, 1, 2])
    rep = classification_report(y_te, yp, labels=[0, 1, 2], target_names=cls_names, output_dict=True, zero_division=0)

    plot_cm(cm, f"SVM RBF ({name})\nAcc={acc*100:.1f}% | F1={f1m*100:.1f}%", fig_dir / f"matriz_confusion_svm_{slug}.png")
    joblib.dump({"pipeline": best_pipe, "classes": cls_names, "variant": name, "params": best_cfg}, mod_dir / f"svm_{slug}.joblib")

    return {
        "variant": name,
        "best_params": best_cfg,
        "train_time_s": round(tune_s, 3),
        "infer_ms": round(inf_ms, 4),
        "val_accuracy": round(best_acc, 4),
        "val_f1_macro": round(best_f1, 4),
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


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Entrena y evalua SVM RBF.")
    parser.add_argument("--pca-hog", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    mod_dir = root / "models" / "svm"
    fig_dir = root / "results" / "figures"
    met_dir = root / "results" / "metrics"
    mod_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(root, rebuild=args.rebuild_cache)
    x_hog, x_ms, y, sps = data["x_hog"], data["x_ms"], data["y"], data["splits"]
    m_tr, m_va, m_te = sps == "train", sps == "val", sps == "test"

    res = {}
    res["solo_hog"] = tune_svm("Solo HOG", x_hog[m_tr], y[m_tr], x_hog[m_va], y[m_va], x_hog[m_te], y[m_te], mod_dir, fig_dir, "hog", args.pca_hog)
    res["multiespectral"] = tune_svm("Multiespectral + Color", x_ms[m_tr], y[m_tr], x_ms[m_va], y[m_va], x_ms[m_te], y[m_te], mod_dir, fig_dir, "multiespectral", 0)

    sc_h = StandardScaler().fit(x_hog[m_tr])
    n_c = min(args.pca_hog, int(m_tr.sum()) - 1, x_hog.shape[1])
    pca = PCA(n_components=n_c, random_state=42).fit(sc_h.transform(x_hog[m_tr]))
    xf_tr = np.hstack([pca.transform(sc_h.transform(x_hog[m_tr])), x_ms[m_tr]])
    xf_va = np.hstack([pca.transform(sc_h.transform(x_hog[m_va])), x_ms[m_va]])
    xf_te = np.hstack([pca.transform(sc_h.transform(x_hog[m_te])), x_ms[m_te]])

    res["fusion"] = tune_svm("Fusion HOG + Multiespectral", xf_tr, y[m_tr], xf_va, y[m_va], xf_te, y[m_te], mod_dir, fig_dir, "fusion", 0)
    joblib.dump({"scaler_hog": sc_h, "pca_hog": pca}, mod_dir / "fusion_transforms_svm.joblib")

    out = {
        "model": "SVC(kernel='rbf', class_weight='balanced')",
        "split_counts": {"train": int(m_tr.sum()), "val": int(m_va.sum()), "test": int(m_te.sum())},
        "variants": res,
    }
    (met_dir / "metricas_test_svm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key, val in res.items():
        print(f"SVM [{key}] Acc={val['test_accuracy']*100:.2f}% | F1={val['test_f1_macro']*100:.2f}%")


if __name__ == "__main__":
    main()
