import argparse
import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_features import cls_names, load_features, plot_cm


def tune_knn(
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
    for k_val in [3, 5, 7, 9, 11, 15]:
        for wt in ["uniform", "distance"]:
            for met in ["euclidean", "manhattan"]:
                steps = [("scaler", StandardScaler())]
                if pca_n > 0:
                    n_c = min(pca_n, len(y_tr) - 1, x_tr.shape[1])
                    steps.append(("pca", PCA(n_components=n_c, random_state=42)))
                steps.append(("knn", KNeighborsClassifier(n_neighbors=k_val, weights=wt, metric=met, n_jobs=-1)))
                pipe = Pipeline(steps)
                pipe.fit(x_tr, y_tr)
                yv = pipe.predict(x_va)
                vf1 = float(f1_score(y_va, yv, average="macro"))
                if vf1 > best_f1:
                    best_f1 = vf1
                    best_acc = float(accuracy_score(y_va, yv))
                    best_cfg = {"n_neighbors": k_val, "weights": wt, "metric": met, "pca": pca_n}
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

    plot_cm(cm, f"KNN ({name})\nAcc={acc*100:.1f}% | F1={f1m*100:.1f}%", fig_dir / f"matriz_confusion_knn_{slug}.png")
    joblib.dump({"pipeline": best_pipe, "classes": cls_names, "variant": name, "params": best_cfg}, mod_dir / f"knn_{slug}.joblib")

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
    parser = argparse.ArgumentParser(description="Entrena y evalua KNN.")
    parser.add_argument("--pca-hog", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    mod_dir = root / "models" / "knn"
    fig_dir = root / "results" / "figures"
    met_dir = root / "results" / "metrics"
    mod_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(root, rebuild=args.rebuild_cache)
    x_hog, x_ms, y, sps = data["x_hog"], data["x_ms"], data["y"], data["splits"]
    m_tr, m_va, m_te = sps == "train", sps == "val", sps == "test"

    res = {}
    # baseline solo hog k=5
    pipe_base = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1))])
    t0 = time.time()
    pipe_base.fit(x_hog[m_tr], y[m_tr])
    yp_b = pipe_base.predict(x_hog[m_te])
    acc_b = float(accuracy_score(y[m_te], yp_b))
    f1_b = float(f1_score(y[m_te], yp_b, average="macro"))
    cm_b = confusion_matrix(y[m_te], yp_b, labels=[0, 1, 2])
    rep_b = classification_report(y[m_te], yp_b, labels=[0, 1, 2], target_names=cls_names, output_dict=True, zero_division=0)
    plot_cm(cm_b, f"KNN Solo HOG\nAcc={acc_b*100:.1f}% | F1={f1_b*100:.1f}%", fig_dir / "matriz_confusion_knn_hog.png")
    joblib.dump({"pipeline": pipe_base, "classes": cls_names, "variant": "solo_hog"}, mod_dir / "knn_hog.joblib")

    res["solo_hog_k5"] = {
        "variant": "solo_hog_k5",
        "best_params": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean", "pca": 0},
        "train_time_s": round(time.time() - t0, 3),
        "test_accuracy": round(acc_b, 4),
        "test_f1_macro": round(f1_b, 4),
        "test_per_class": {cn: {"f1": round(float(rep_b[cn]["f1-score"]), 4)} for cn in cls_names},
        "confusion_matrix": cm_b.tolist(),
    }

    res["multiespectral"] = tune_knn(
        "Multiespectral + Color", x_ms[m_tr], y[m_tr], x_ms[m_va], y[m_va], x_ms[m_te], y[m_te],
        mod_dir, fig_dir, "multiespectral", 0,
    )

    sc_h = StandardScaler().fit(x_hog[m_tr])
    n_c = min(args.pca_hog, int(m_tr.sum()) - 1, x_hog.shape[1])
    pca = PCA(n_components=n_c, random_state=42).fit(sc_h.transform(x_hog[m_tr]))
    xf_tr = np.hstack([pca.transform(sc_h.transform(x_hog[m_tr])), x_ms[m_tr]])
    xf_va = np.hstack([pca.transform(sc_h.transform(x_hog[m_va])), x_ms[m_va]])
    xf_te = np.hstack([pca.transform(sc_h.transform(x_hog[m_te])), x_ms[m_te]])

    res["fusion"] = tune_knn(
        "Fusion HOG + Multiespectral", xf_tr, y[m_tr], xf_va, y[m_va], xf_te, y[m_te],
        mod_dir, fig_dir, "fusion", 0,
    )
    joblib.dump({"scaler_hog": sc_h, "pca_hog": pca}, mod_dir / "fusion_transforms_knn.joblib")

    out = {
        "model": "KNeighborsClassifier",
        "split_counts": {"train": int(m_tr.sum()), "val": int(m_va.sum()), "test": int(m_te.sum())},
        "variants": res,
    }
    (met_dir / "metricas_test_knn.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key, val in res.items():
        print(f"KNN [{key}] Acc={val['test_accuracy']*100:.2f}% | F1={val['test_f1_macro']*100:.2f}%")


if __name__ == "__main__":
    main()
