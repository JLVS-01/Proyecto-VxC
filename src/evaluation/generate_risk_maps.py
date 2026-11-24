import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


cls_rgb = {
    0: (225, 45, 45),
    1: (45, 175, 70),
    2: (190, 195, 205),
}


def win_starts(length: int, win: int, stride: int) -> List[int]:
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, stride))
    if starts[-1] < length - win:
        starts.append(length - win)
    return starts


def slide_segment(
    model: YOLO,
    bgr: np.ndarray,
    win_sz: int = 1024,
    overlap: float = 0.25,
    imgsz: int = 512,
    conf: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray, int]:
    # inferencia por ventanas deslizantes con ponderacion hann
    rows, cols = bgr.shape[:2]
    stride = int(round(win_sz * (1.0 - overlap)))
    xs = win_starts(cols, win_sz, stride)
    ys = win_starts(rows, win_sz, stride)

    wy = np.maximum(np.hanning(win_sz).astype(np.float32), 0.15)
    wx = np.maximum(np.hanning(win_sz).astype(np.float32), 0.15)
    w2d = np.outer(wy, wx)

    p_acc = np.zeros((3, rows, cols), dtype=np.float32)
    w_acc = np.zeros((rows, cols), dtype=np.float32)
    n_win = 0

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + win_sz, rows), min(x0 + win_sz, cols)
            crop = bgr[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]

            res = model.predict(source=crop, imgsz=imgsz, conf=conf, iou=0.5, retina_masks=True, device="cpu", verbose=False)[0]
            l_sc = np.zeros((3, ch, cw), dtype=np.float32)
            if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
                m_arr = res.masks.data.cpu().numpy()
                c_arr = res.boxes.cls.cpu().numpy().astype(int)
                cf_arr = res.boxes.conf.cpu().numpy().astype(np.float32)
                for msk, cid, cf in zip(m_arr, c_arr, cf_arr):
                    if cid not in (0, 1, 2):
                        continue
                    if msk.shape[0] != ch or msk.shape[1] != cw:
                        msk = cv2.resize(msk, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    l_sc[cid] = np.maximum(l_sc[cid], (msk > 0.5).astype(np.float32) * cf)

            wp = w2d[:ch, :cw]
            for cid in range(3):
                p_acc[cid, y0:y1, x0:x1] += l_sc[cid] * wp
            w_acc[y0:y1, x0:x1] += wp
            n_win += 1

    p_n = p_acc / np.maximum(w_acc, 1e-6)[None, :, :]
    sem = np.argmax(p_n, axis=0).astype(np.uint8)
    sem[p_n.max(axis=0) < 0.08] = 255
    return sem, p_n[0], n_win


def risk_level(pct_dry: float) -> Tuple[str, str]:
    if pct_dry >= 30.0:
        return "ALTO", "#d62828"
    if pct_dry >= 15.0:
        return "MODERADO", "#f77f00"
    return "BAJO", "#2a9d8f"


def save_panel(out_path: Path, bgr: np.ndarray, sem: np.ndarray, dry_p: np.ndarray, st: Dict) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows, cols = rgb.shape[:2]

    cmask = np.zeros_like(rgb)
    for cid, col in cls_rgb.items():
        cmask[sem == cid] = col
    has_c = sem < 3
    ovl = rgb.copy()
    ovl[has_c] = (0.58 * rgb[has_c] + 0.42 * cmask[has_c]).astype(np.uint8)

    tw = 900
    th = int(round(rows * (tw / cols)))
    rgb_s = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    seg_s = cv2.resize(ovl, (tw, th), interpolation=cv2.INTER_AREA)
    dry_s = cv2.resize(dry_p, (tw, th), interpolation=cv2.INTER_AREA)

    hu8 = np.clip(dry_s / 0.65 * 255.0, 0, 255).astype(np.uint8)
    hrgb = cv2.cvtColor(cv2.applyColorMap(hu8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    risk_img = (0.52 * rgb_s + 0.48 * hrgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=180)
    axes[0].imshow(rgb_s)
    axes[0].set_title("Fotografia Aerea Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(seg_s)
    axes[1].set_title(
        f"Segmentacion YOLO11n-seg\nSeca: {st['pct_vegetacion_seca']:.1f}% | Sana: {st['pct_vegetacion_sana']:.1f}% | Suelo: {st['pct_suelo_desnudo']:.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(risk_img)
    axes[2].set_title(f"Mapa de Calor Combustible - RIESGO {st['nivel_riesgo']}", fontsize=11, fontweight="bold", color=st["risk_color"])
    axes[2].axis("off")

    fig.suptitle(f"Waypoint {st['waypoint_id']:02d} ({st['year']}) | Riesgo: {st['nivel_riesgo']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    qa_csv = root / "data" / "processed" / "qa_alineacion.csv"
    sp_csv = root / "data" / "yolo_dataset" / "resumen_split.csv"
    raw_dir = root / "data" / "raw"
    best_pt = root / "models" / "yolo" / "best.pt"
    met_dir = root / "results" / "metrics"
    maps_dir = root / "results" / "risk_maps"

    parser = argparse.ArgumentParser(description="Genera mapas aereos de riesgo de incendio.")
    parser.add_argument("--weights", type=Path, default=best_pt)
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--waypoints", type=int, nargs="*", default=None)
    args = parser.parse_args()

    if args.waypoints:
        wps = set(args.waypoints)
    else:
        wps = set()
        with open(sp_csv, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec["split"] == "test":
                    wps.add(int(rec["waypoint_id"]))

    caps = []
    with open(qa_csv, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if int(rec["waypoint_id"]) in wps:
                caps.append(rec)

    model = YOLO(str(args.weights))
    rows: List[Dict] = []

    for cap in tqdm(caps, desc="Generando mapas"):
        wp = int(cap["waypoint_id"])
        yr = cap["year"]
        cid = cap["capture_id"]
        x0, y0, x1, y1 = [int(val) for val in cap["valid_box"].split(",")]

        full = cv2.imread(str(raw_dir / yr / f"{cid}_D.JPG"), cv2.IMREAD_COLOR)
        crop = full[y0:y1, x0:x1]

        t0 = time.time()
        sem, dry_p, n_win = slide_segment(model, crop, 1024, 0.25, 512, args.conf)
        dt = time.time() - t0

        tot = sem.size
        p_dry = 100.0 * float((sem == 0).sum()) / tot
        p_sana = 100.0 * float((sem == 1).sum()) / tot
        p_soil = 100.0 * float((sem == 2).sum()) / tot
        p_bg = 100.0 * float((sem == 255).sum()) / tot
        lvl, col_hex = risk_level(p_dry)

        st = {
            "waypoint_id": wp,
            "year": yr,
            "capture_id": cid,
            "gps_lat": cap["gps_lat"],
            "gps_lon": cap["gps_lon"],
            "ventanas_eval": n_win,
            "tiempo_s": round(dt, 2),
            "pct_vegetacion_seca": round(p_dry, 2),
            "pct_vegetacion_sana": round(p_sana, 2),
            "pct_suelo_desnudo": round(p_soil, 2),
            "pct_sombra_sin_clase": round(p_bg, 2),
            "nivel_riesgo": lvl,
            "risk_color": col_hex,
        }
        out_img = maps_dir / f"wp{wp:02d}_{yr}_{cid}_mapa_riesgo.jpg"
        save_panel(out_img, crop, sem, dry_p, st)

        clean = {k: v for k, v in st.items() if k != "risk_color"}
        clean["mapa_path"] = str(out_img)
        rows.append(clean)

    rows.sort(key=lambda r: (r["waypoint_id"], r["year"]))
    csv_p = met_dir / "reporte_riesgo_incendio.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"Mapas generados: {len(rows)} | Reporte: {csv_p}")


if __name__ == "__main__":
    main()
