import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


cls_names = ["vegetacion_seca", "vegetacion_sana", "suelo_desnudo"]

ms_names = [
    "ndvi_mean",
    "ndvi_std",
    "ndvi_p10",
    "ndvi_p90",
    "ndre_mean",
    "ndre_std",
    "gndvi_mean",
    "exg_mean",
    "exg_std",
    "hsv_h_mean",
    "hsv_s_mean",
    "hsv_s_std",
    "hsv_v_mean",
    "hsv_v_std",
    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",
    "lab_b_std",
    "sobel_mean",
    "sobel_std",
    "sobel_p90",
    "area_log",
]

patch_sz = (128, 128)
hog_cfg = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def extract_tile_feats(
    row: Dict,
    yolo_dir: Path,
    proc_dir: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str], List[int]]:
    tid = row["tile_id"]
    sp = row["split"]
    yr = row["year"]
    wp = int(row["waypoint_id"])

    img_p = yolo_dir / "images" / sp / f"{tid}.jpg"
    lbl_p = yolo_dir / "labels" / sp / f"{tid}.txt"
    idx_p = proc_dir / "tiles_indices" / yr / f"{tid}.npz"

    if not lbl_p.exists():
        return [], [], [], [], []
    lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return [], [], [], [], []

    bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    rows, cols = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    rf = rgb[:, :, 0].astype(np.float32)
    gf = rgb[:, :, 1].astype(np.float32)
    bf = rgb[:, :, 2].astype(np.float32)
    exg = (2.0 * gf - rf - bf) / (rf + gf + bf + 1.0)

    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    sob = np.hypot(gx, gy)

    npz = np.load(str(idx_p))
    ndvi = cv2.resize(npz["ndvi"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    ndre = cv2.resize(npz["ndre"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    gndvi = cv2.resize(npz["gndvi"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)

    hogs: List[np.ndarray] = []
    mss: List[np.ndarray] = []
    ys: List[int] = []
    sps: List[str] = []
    wps: List[int] = []

    for ln in lines:
        parts = ln.split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in (0, 1, 2):
            continue
        coords = [float(val) for val in parts[1:]]
        pts = np.array(
            [[coords[i] * cols, coords[i + 1] * rows] for i in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        x0 = max(0, int(pts[:, 0].min()))
        y0 = max(0, int(pts[:, 1].min()))
        x1 = min(cols, int(pts[:, 0].max()) + 1)
        y1 = min(rows, int(pts[:, 1].max()) + 1)
        if (x1 - x0) < 12 or (y1 - y0) < 12:
            continue

        loc_pts = pts - np.array([[x0, y0]], dtype=np.int32)
        pmask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(pmask, [loc_pts.reshape((-1, 1, 2))], 255)
        msk = pmask > 0
        area = int(msk.sum())
        if area < 100:
            continue

        # vector hog en escala de grises
        patch = cv2.resize(gray[y0:y1, x0:x1], patch_sz, interpolation=cv2.INTER_AREA)
        h_vec = hog(
            patch,
            orientations=hog_cfg["orientations"],
            pixels_per_cell=hog_cfg["pixels_per_cell"],
            cells_per_block=hog_cfg["cells_per_block"],
            block_norm=hog_cfg["block_norm"],
            feature_vector=True,
        ).astype(np.float32)

        # rasgos espectrales y textura dentro de la mascara
        nv = ndvi[y0:y1, x0:x1][msk]
        rev = ndre[y0:y1, x0:x1][msk]
        gv = gndvi[y0:y1, x0:x1][msk]
        ev = exg[y0:y1, x0:x1][msk]
        hsv_s = hsv[y0:y1, x0:x1]
        lab_s = lab[y0:y1, x0:x1]
        sv = sob[y0:y1, x0:x1][msk]

        m_vec = np.array(
            [
                float(nv.mean()),
                float(nv.std()),
                float(np.percentile(nv, 10)),
                float(np.percentile(nv, 90)),
                float(rev.mean()),
                float(rev.std()),
                float(gv.mean()),
                float(ev.mean()),
                float(ev.std()),
                float(hsv_s[:, :, 0][msk].mean()),
                float(hsv_s[:, :, 1][msk].mean()),
                float(hsv_s[:, :, 1][msk].std()),
                float(hsv_s[:, :, 2][msk].mean()),
                float(hsv_s[:, :, 2][msk].std()),
                float(lab_s[:, :, 0][msk].mean()),
                float(lab_s[:, :, 1][msk].mean()),
                float(lab_s[:, :, 2][msk].mean()),
                float(lab_s[:, :, 2][msk].std()),
                float(sv.mean()),
                float(sv.std()),
                float(np.percentile(sv, 90)),
                float(np.log1p(area)),
            ],
            dtype=np.float32,
        )

        hogs.append(h_vec)
        mss.append(m_vec)
        ys.append(cid)
        sps.append(sp)
        wps.append(wp)

    return hogs, mss, ys, sps, wps


def load_features(root: Path, rebuild: bool = False, workers: int = 0) -> Dict[str, np.ndarray]:
    cache_p = root / "data" / "processed" / "cache_features_clasicos.npz"
    if cache_p.exists() and not rebuild:
        data = np.load(str(cache_p), allow_pickle=False)
        return {key: data[key] for key in data.files}

    proc_dir = root / "data" / "processed"
    yolo_dir = root / "data" / "yolo_dataset"
    with open(yolo_dir / "resumen_split.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if workers <= 0:
        workers = max(1, min(6, (os.cpu_count() or 4) // 2))

    all_hog: List[np.ndarray] = []
    all_ms: List[np.ndarray] = []
    all_y: List[int] = []
    all_sp: List[str] = []
    all_wp: List[int] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(extract_tile_feats, rec, yolo_dir, proc_dir) for rec in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extrayendo rasgos"):
            hogs, mss, ys, sps, wps = fut.result()
            all_hog.extend(hogs)
            all_ms.extend(mss)
            all_y.extend(ys)
            all_sp.extend(sps)
            all_wp.extend(wps)

    out = {
        "x_hog": np.vstack(all_hog).astype(np.float32),
        "x_ms": np.vstack(all_ms).astype(np.float32),
        "y": np.array(all_y, dtype=np.int32),
        "splits": np.array(all_sp, dtype="<U8"),
        "waypoints": np.array(all_wp, dtype=np.int32),
    }
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_p), **out)
    return out


def plot_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
    cm_n = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporcion por clase real")

    ax.set_xticks(range(len(cls_names)))
    ax.set_yticks(range(len(cls_names)))
    ax.set_xticklabels(cls_names, rotation=18, ha="right", fontsize=10)
    ax.set_yticklabels(cls_names, fontsize=10)
    ax.set_xlabel("Prediccion del modelo", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clase real", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold")

    for i in range(len(cls_names)):
        for j in range(len(cls_names)):
            val = cm_n[i, j]
            col = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val*100:.1f}%\n({int(cm[i, j])})", ha="center", va="center", color=col, fontsize=9.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
