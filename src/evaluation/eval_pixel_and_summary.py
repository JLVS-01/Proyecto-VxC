import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


cls_names = ["vegetacion_seca", "vegetacion_sana", "suelo_desnudo"]


def raster_gt(lbl_p: Path, rows: int, cols: int) -> np.ndarray:
    mask = np.full((rows, cols), 255, dtype=np.uint8)
    if not lbl_p.exists():
        return mask
    txt = lbl_p.read_text(encoding="utf-8").strip()
    if not txt:
        return mask

    polys: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    for ln in txt.splitlines():
        parts = ln.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in polys:
            continue
        coords = [float(val) for val in parts[1:]]
        pts = np.array([[coords[i] * cols, coords[i + 1] * rows] for i in range(0, len(coords) - 1, 2)], dtype=np.int32)
        polys[cid].append(pts)

    for cid in (1, 0, 2):
        for pts in polys[cid]:
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], cid)
    return mask


def pred_mask(model: YOLO, img_p: Path, imgsz: int = 512, conf: float = 0.20) -> np.ndarray:
    bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    rows, cols = bgr.shape[:2]
    res = model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=0.5, retina_masks=True, device="cpu", verbose=False)[0]

    smap = np.zeros((3, rows, cols), dtype=np.float32)
    if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
        m_arr = res.masks.data.cpu().numpy()
        c_arr = res.boxes.cls.cpu().numpy().astype(int)
        cf_arr = res.boxes.conf.cpu().numpy().astype(np.float32)
        for msk, cid, cf in zip(m_arr, c_arr, cf_arr):
            if cid not in (0, 1, 2):
                continue
            if msk.shape[0] != rows or msk.shape[1] != cols:
                msk = cv2.resize(msk, (cols, rows), interpolation=cv2.INTER_LINEAR)
            smap[cid] = np.maximum(smap[cid], (msk > 0.5).astype(np.float32) * cf)

    max_s = smap.max(axis=0)
    pred = np.argmax(smap, axis=0).astype(np.uint8)
    pred[max_s <= 0.0] = 255
    return pred


def plot_px_cm(cm: np.ndarray, acc: float, miou: float, out_path: Path) -> None:
    cm_n = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporcion de pixeles")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(cls_names, rotation=18, ha="right", fontsize=10)
    ax.set_yticklabels(cls_names, fontsize=10)
    ax.set_xlabel("Prediccion YOLO11n-seg", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clase real", fontsize=11, fontweight="bold")
    ax.set_title(f"Matriz de Confusion Pixel - YOLO11n-seg\nAcc={acc*100:.1f}% | mIoU={miou*100:.1f}%", fontsize=11, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = cm_n[i, j]
            col = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val*100:.1f}%\n({cm[i, j]/1e6:.2f}M px)", ha="center", va="center", color=col, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_master(man_m: Dict, y_px: Dict, y_inst: Dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), dpi=300)

    mods = ["KNN", "Random\nForest", "SVM\n(RBF)"]
    keys = ["knn", "rf", "svm"]
    f1_h = [man_m["trained_on_jorge_manual"]["solo_hog"][k]["f1_macro"] * 100.0 for k in keys]
    f1_m = [man_m["trained_on_jorge_manual"]["multiespectral"][k]["f1_macro"] * 100.0 for k in keys]
    f1_z = [man_m["zero_shot_to_manual"][k]["f1_macro"] * 100.0 for k in keys]

    xp = np.arange(len(mods))
    wd = 0.25
    b1 = axes[0].bar(xp - wd, f1_h, wd, label="Solo HOG", color="#8d99ae")
    b2 = axes[0].bar(xp, f1_z, wd, label="Zero-Shot", color="#e9c46a")
    b3 = axes[0].bar(xp + wd, f1_m, wd, label="Multiespectral + Color", color="#2a9d8f")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(mods, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 128)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel("F1-Macro (%)", fontsize=10.5, fontweight="bold")
    axes[0].set_title("Modelos Clasicos (Multiespectral)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper center", ncol=1, fontsize=8.5, framealpha=0.95)
    for bars in (b1, b2, b3):
        for bar in bars:
            hv = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, hv + 1.4, f"{hv:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    xp2 = np.arange(len(cls_names))
    pf1 = [y_px["per_class"][c]["pixel_f1"] * 100.0 for c in cls_names]
    piou = [y_px["per_class"][c]["pixel_iou"] * 100.0 for c in cls_names]
    imap = [y_inst["test_per_class"][c]["mask_mAP50"] * 100.0 for c in cls_names]

    c1 = axes[1].bar(xp2 - wd, pf1, wd, label="F1 Pixel", color="#264653")
    c2 = axes[1].bar(xp2, piou, wd, label="IoU Pixel", color="#457b9d")
    c3 = axes[1].bar(xp2 + wd, imap, wd, label="mAP50 Mascara", color="#e76f51")
    axes[1].set_xticks(xp2)
    axes[1].set_xticklabels(["vegetacion\nseca", "vegetacion\nsana", "suelo\ndesnudo"], fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 128)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])
    axes[1].set_ylabel("Porcentaje (%)", fontsize=10.5, fontweight="bold")
    axes[1].set_title(f"YOLO11n-seg Solo RGB | Acc={y_px['pixel_accuracy']*100:.1f}% | mIoU={y_px['mIoU']*100:.1f}%", fontsize=10.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    for bars in (c1, c2, c3):
        for bar in bars:
            hv = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, hv + 1.4, f"{hv:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Comparativa Integral de Modelos - Split Espacial", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    img_dir = root / "data" / "yolo_dataset" / "images" / "test"
    lbl_dir = root / "data" / "yolo_dataset" / "labels" / "test"
    met_dir = root / "results" / "metrics"
    fig_dir = root / "results" / "figures"
    best_pt = root / "models" / "yolo" / "best.pt"
    met_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_pt))
    paths = sorted(img_dir.glob("*.jpg"))

    cm = np.zeros((3, 3), dtype=np.int64)
    inter = np.zeros(3, dtype=np.int64)
    union = np.zeros(3, dtype=np.int64)
    gt_tot = np.zeros(3, dtype=np.int64)
    pr_tot = np.zeros(3, dtype=np.int64)
    cov_px = 0
    all_px = 0

    t0 = time.time()
    for ip in tqdm(paths, desc="Evaluando YOLO en test"):
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        rows, cols = bgr.shape[:2]
        gt = raster_gt(lbl_dir / f"{ip.stem}.txt", rows, cols)
        pr = pred_mask(model, ip, imgsz=512, conf=0.20)

        vgt = gt < 3
        all_px += int(vgt.sum())
        cov_px += int((vgt & (pr < 3)).sum())

        for ci in range(3):
            gi = gt == ci
            pi = pr == ci
            gt_tot[ci] += int(gi.sum())
            pr_tot[ci] += int((pi & vgt).sum())
            inter[ci] += int((gi & pi).sum())
            union[ci] += int((gi | (pi & vgt)).sum())
            for cj in range(3):
                cm[ci, cj] += int((gi & (pr == cj)).sum())

    dt = time.time() - t0
    px_acc = float(np.trace(cm) / max(int(cm.sum()), 1))
    per_cls = {}
    f1s, ious = [], []
    for ci, cn in enumerate(cls_names):
        tp = int(inter[ci])
        prec = float(tp / max(int(pr_tot[ci]), 1))
        rec = float(tp / max(int(gt_tot[ci]), 1))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-9))
        iou = float(tp / max(int(union[ci]), 1))
        f1s.append(f1)
        ious.append(iou)
        per_cls[cn] = {"pixel_precision": round(prec, 4), "pixel_recall": round(rec, 4), "pixel_f1": round(f1, 4), "pixel_iou": round(iou, 4)}

    y_px = {
        "num_test_tiles": len(paths),
        "eval_time_s": round(dt, 2),
        "gt_coverage_ratio": round(float(cov_px / max(all_px, 1)), 4),
        "pixel_accuracy": round(px_acc, 4),
        "pixel_f1_macro": round(float(np.mean(f1s)), 4),
        "mIoU": round(float(np.mean(ious)), 4),
        "per_class": per_cls,
        "confusion_matrix_px": cm.tolist(),
    }
    (met_dir / "metricas_pixel_yolo.json").write_text(json.dumps(y_px, indent=2), encoding="utf-8")
    plot_px_cm(cm, px_acc, float(np.mean(ious)), fig_dir / "matriz_confusion_pixel_yolo.png")

    man_m = json.loads((met_dir / "metricas_jorge_manual.json").read_text(encoding="utf-8"))
    y_inst = json.loads((met_dir / "metricas_test_yolo.json").read_text(encoding="utf-8"))
    plot_master(man_m, y_px, y_inst, fig_dir / "comparativa_maestra_ieee.png")

    print(f"Pixel Acc={px_acc*100:.2f}% | F1={y_px['pixel_f1_macro']*100:.2f}% | mIoU={y_px['mIoU']*100:.2f}%")


if __name__ == "__main__":
    main()
