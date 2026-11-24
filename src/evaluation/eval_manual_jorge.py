import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from common_features import cls_names, hog_cfg, patch_sz, plot_cm


cls_to_id = {name: idx for idx, name in enumerate(cls_names)}


def read_dewarp_h(tif_p: Path) -> np.ndarray:
    with open(tif_p, "rb") as fh:
        raw = fh.read(180000)
    match = re.search(rb"DewarpHMatrix[=>\"]+([^<\"\n]+)", raw)
    if match:
        vals = [float(val) for val in match.group(1).decode().split(",")]
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    return np.array([[1.7162, 0.0, 415.752], [0.0, 1.7162, 309.814], [0.0, 0.0, 1.0]], dtype=np.float64)


def norm_u8(img: np.ndarray) -> np.ndarray:
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)


def align_ms_rgb(rgb_g: np.ndarray, ms_g: np.ndarray, fb_h: np.ndarray) -> np.ndarray:
    r_sm = norm_u8(cv2.resize(rgb_g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    m_sm = norm_u8(cv2.resize(ms_g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    sift = cv2.SIFT_create(nfeatures=3500)
    kp1, d1 = sift.detectAndCompute(m_sm, None)
    kp2, d2 = sift.detectAndCompute(r_sm, None)
    if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
        return fb_h
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m1 for m1, m2 in pairs if m1.distance < 0.72 * m2.distance]
    if len(good) < 20:
        return fb_h
    pts_m = np.float32([kp1[m1.queryIdx].pt for m1 in good]) / 0.5
    pts_r = np.float32([kp2[m1.trainIdx].pt for m1 in good]) / 0.25
    h_mat, mask = cv2.findHomography(pts_m, pts_r, cv2.RANSAC, 6.0)
    if h_mat is None or mask is None or int(mask.sum()) < 20:
        return fb_h
    return h_mat


def extract_manual_json(
    json_p: Path,
    cap_wp: Dict[str, int],
    raw_dir: Path,
    mb_dir: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    cap_id = json_p.stem.replace("_D", "")
    yr = "2025" if "2025" in cap_id else "2026"
    wp = cap_wp.get(cap_id, -1)

    rgb_p = raw_dir / yr / f"{cap_id}_D.JPG"
    mb_p = mb_dir / yr / f"wp{wp:02d}_{cap_id}_MS_aligned.tif"
    g_tif = raw_dir / yr / f"{cap_id}_MS_G.TIF"
    if not rgb_p.exists() or not mb_p.exists():
        return [], [], [], []

    shapes = json.loads(json_p.read_text(encoding="utf-8")).get("shapes", [])
    if not shapes:
        return [], [], [], []

    bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
    rows, cols = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    rf, gf, bf = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    exg = (2.0 * gf - rf - bf) / (rf + gf + bf + 1.0)
    sob = np.hypot(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))

    mb = tifffile.imread(str(mb_p)).astype(np.float32)
    g_ms, r_ms, re_ms, nir_ms = mb[0], mb[1], mb[2], mb[3]
    eps = 1e-6
    ndvi_ms = np.clip((nir_ms - r_ms) / (nir_ms + r_ms + eps), -1.0, 1.0)
    ndre_ms = np.clip((nir_ms - re_ms) / (nir_ms + re_ms + eps), -1.0, 1.0)
    gndvi_ms = np.clip((nir_ms - g_ms) / (nir_ms + g_ms + eps), -1.0, 1.0)

    fb_h = read_dewarp_h(g_tif) if g_tif.exists() else np.eye(3)
    h_mat = align_ms_rgb(bgr[:, :, 1].astype(np.float32), g_ms, fb_h)

    ndvi_r = cv2.warpPerspective(ndvi_ms, h_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    ndre_r = cv2.warpPerspective(ndre_ms, h_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    gndvi_r = cv2.warpPerspective(gndvi_ms, h_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    val_r = cv2.warpPerspective(np.ones_like(g_ms, dtype=np.uint8), h_mat, (cols, rows), flags=cv2.INTER_NEAREST) > 0

    hogs, mss, ys, wps = [], [], [], []
    for sh in shapes:
        lbl = sh.get("label", "").strip()
        if lbl not in cls_to_id:
            continue
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue
        xs = [pt[0] for pt in pts]
        ys_p = [pt[1] for pt in pts]
        x0, x1 = max(0, int(round(min(xs)))), min(cols, int(round(max(xs))))
        y0, y1 = max(0, int(round(min(ys_p)))), min(rows, int(round(max(ys_p))))
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            continue

        vm = val_r[y0:y1, x0:x1]
        if float(vm.mean()) < 0.50:
            continue

        patch = cv2.resize(gray[y0:y1, x0:x1], patch_sz, interpolation=cv2.INTER_AREA)
        h_vec = hog(
            patch,
            orientations=hog_cfg["orientations"],
            pixels_per_cell=hog_cfg["pixels_per_cell"],
            cells_per_block=hog_cfg["cells_per_block"],
            block_norm=hog_cfg["block_norm"],
            feature_vector=True,
        ).astype(np.float32)

        nv = ndvi_r[y0:y1, x0:x1][vm]
        rev = ndre_r[y0:y1, x0:x1][vm]
        gv = gndvi_r[y0:y1, x0:x1][vm]
        ev = exg[y0:y1, x0:x1][vm]
        sv = sob[y0:y1, x0:x1][vm]

        m_vec = np.array(
            [
                float(nv.mean()), float(nv.std()), float(np.percentile(nv, 10)), float(np.percentile(nv, 90)),
                float(rev.mean()), float(rev.std()), float(gv.mean()),
                float(ev.mean()), float(ev.std()),
                float(hsv[y0:y1, x0:x1, 0][vm].mean()), float(hsv[y0:y1, x0:x1, 1][vm].mean()), float(hsv[y0:y1, x0:x1, 1][vm].std()),
                float(hsv[y0:y1, x0:x1, 2][vm].mean()), float(hsv[y0:y1, x0:x1, 2][vm].std()),
                float(lab[y0:y1, x0:x1, 0][vm].mean()), float(lab[y0:y1, x0:x1, 1][vm].mean()),
                float(lab[y0:y1, x0:x1, 2][vm].mean()), float(lab[y0:y1, x0:x1, 2][vm].std()),
                float(sv.mean()), float(sv.std()), float(np.percentile(sv, 90)),
                float(np.log1p(int(vm.sum()))),
            ],
            dtype=np.float32,
        )

        hogs.append(h_vec)
        mss.append(m_vec)
        ys.append(cls_to_id[lbl])
        wps.append(wp)

    return hogs, mss, ys, wps


def plot_comp_bar(res_man: Dict, out_path: Path) -> None:
    mods = ["KNN", "Random Forest", "SVM (RBF)"]
    keys = ["knn", "rf", "svm"]
    acc_h = [res_man["solo_hog"][k]["accuracy"] * 100.0 for k in keys]
    acc_m = [res_man["multiespectral"][k]["accuracy"] * 100.0 for k in keys]
    f1_h = [res_man["solo_hog"][k]["f1_macro"] * 100.0 for k in keys]
    f1_m = [res_man["multiespectral"][k]["f1_macro"] * 100.0 for k in keys]

    xpos = np.arange(len(mods))
    wd = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300, sharey=True)

    for ax, vh, vm, ttl in [(axes[0], acc_h, acc_m, "Exactitud (%)"), (axes[1], f1_h, f1_m, "F1-Macro (%)")]:
        b1 = ax.bar(xpos - wd / 2, vh, wd, label="Solo HOG", color="#8d99ae")
        b2 = ax.bar(xpos + wd / 2, vm, wd, label="Multiespectral + Color", color="#2a9d8f")
        ax.set_xticks(xpos)
        ax.set_xticklabels(mods, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_title(ttl, fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bars in (b1, b2):
            for bar in bars:
                hv = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, hv + 1.5, f"{hv:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_ylabel("Porcentaje en test espacial (%)", fontsize=10.5, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.suptitle("Comparativa en Anotaciones Manuales (Split Espacial)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    man_dir = root / "data" / "annotations" / "manual_jorge"
    raw_dir = root / "data" / "raw"
    mb_dir = root / "data" / "processed" / "multibanda"
    qa_csv = root / "data" / "processed" / "qa_alineacion.csv"
    sp_csv = root / "data" / "yolo_dataset" / "resumen_split.csv"
    fig_dir = root / "results" / "figures"
    met_dir = root / "results" / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    cap_wp: Dict[str, int] = {}
    with open(qa_csv, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            cap_wp[rec["capture_id"]] = int(rec["waypoint_id"])

    wp_sp: Dict[int, str] = {}
    with open(sp_csv, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            wp_sp[int(rec["waypoint_id"])] = rec["split"]

    cache_p = root / "data" / "processed" / "cache_features_manual.npz"
    if cache_p.exists():
        dat = np.load(str(cache_p), allow_pickle=False)
        x_hog, x_ms, y, wps = dat["x_hog"], dat["x_ms"], dat["y"], dat["waypoints"]
    else:
        jfiles = sorted(man_dir.glob("*.json"))
        all_h, all_m, all_y, all_w = [], [], [], []
        wk = max(1, min(6, (os.cpu_count() or 4) // 2))
        with ProcessPoolExecutor(max_workers=wk) as pool:
            futs = [pool.submit(extract_manual_json, jf, cap_wp, raw_dir, mb_dir) for jf in jfiles]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Procesando manuales"):
                hl, ml, yl, wl = fut.result()
                all_h.extend(hl)
                all_m.extend(ml)
                all_y.extend(yl)
                all_w.extend(wl)
        x_hog = np.vstack(all_h).astype(np.float32)
        x_ms = np.vstack(all_m).astype(np.float32)
        y = np.array(all_y, dtype=np.int32)
        wps = np.array(all_w, dtype=np.int32)
        np.savez_compressed(str(cache_p), x_hog=x_hog, x_ms=x_ms, y=y, waypoints=wps)

    sps = np.array([wp_sp.get(int(w), "train") for w in wps])
    m_tr = sps == "train"
    m_te = (sps == "val") | (sps == "test")

    def run_suite(x_tr, y_tr, x_te, y_te, tag: str, do_pca: bool) -> Dict:
        n_c = min(48, len(y_tr) - 1)
        pca_step = [("pca", PCA(n_components=n_c, random_state=42))] if do_pca else []
        mods = {
            "knn": Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan"))]),
            "rf": RandomForestClassifier(n_estimators=300, max_depth=18, class_weight="balanced", n_jobs=-1, random_state=42),
            "svm": Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", random_state=42))]),
        }
        out_m = {}
        for mk, clf in mods.items():
            clf.fit(x_tr, y_tr)
            yp = clf.predict(x_te)
            acc = float(accuracy_score(y_te, yp))
            f1m = float(f1_score(y_te, yp, average="macro"))
            cm = confusion_matrix(y_te, yp, labels=[0, 1, 2])
            rep = classification_report(y_te, yp, labels=[0, 1, 2], target_names=cls_names, output_dict=True, zero_division=0)
            if tag == "multiespectral":
                plot_cm(cm, f"{mk.upper()} Multiespectral (Manual)\nAcc={acc*100:.1f}% | F1={f1m*100:.1f}%", fig_dir / f"matriz_confusion_manual_{mk}_ms.png")
            out_m[mk] = {
                "accuracy": round(acc, 4),
                "f1_macro": round(f1m, 4),
                "per_class_f1": {cn: round(float(rep[cn]["f1-score"]), 4) for cn in cls_names},
                "confusion_matrix": cm.tolist(),
            }
        return out_m

    res_man = {
        "solo_hog": run_suite(x_hog[m_tr], y[m_tr], x_hog[m_te], y[m_te], "solo_hog", True),
        "multiespectral": run_suite(x_ms[m_tr], y[m_tr], x_ms[m_te], y[m_te], "multiespectral", False),
    }

    zshot = {}
    for mk, rel, kobj in [
        ("knn", "knn/knn_multiespectral.joblib", "pipeline"),
        ("rf", "random_forest/rf_multiespectral.joblib", "model"),
        ("svm", "svm/svm_multiespectral.joblib", "pipeline"),
    ]:
        mp = root / "models" / rel
        if mp.exists():
            clf = joblib.load(mp)[kobj]
            yp = clf.predict(x_ms[m_te])
            rep = classification_report(y[m_te], yp, labels=[0, 1, 2], target_names=cls_names, output_dict=True, zero_division=0)
            zshot[mk] = {
                "accuracy": round(float(accuracy_score(y[m_te], yp)), 4),
                "f1_macro": round(float(f1_score(y[m_te], yp, average="macro")), 4),
                "per_class_f1": {cn: round(float(rep[cn]["f1-score"]), 4) for cn in cls_names},
            }

    plot_comp_bar(res_man, fig_dir / "comparativa_modelos_ieee.png")
    full = {
        "total_manual_boxes_valid": int(len(y)),
        "train_boxes": int(m_tr.sum()),
        "test_spatial_boxes": int(m_te.sum()),
        "trained_on_jorge_manual": res_man,
        "zero_shot_to_manual": zshot,
    }
    (met_dir / "metricas_jorge_manual.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"Evaluacion manual completa: {len(y)} cajas validas")


if __name__ == "__main__":
    main()
