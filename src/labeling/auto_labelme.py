import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.segmentation import slic
from tqdm import tqdm


cls_colors = {
    "vegetacion_seca": (30, 30, 235),
    "vegetacion_sana": (40, 190, 40),
    "suelo_desnudo": (215, 195, 140),
}


def extract_sp_feats(
    bgr: np.ndarray,
    ndvi: np.ndarray,
    ndre: np.ndarray,
    gndvi: np.ndarray,
    n_seg: int = 450,
) -> Tuple[np.ndarray, List[Dict]]:
    # extrae rasgos espectrales y textura por superpixel slic
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rf = rgb[:, :, 0].astype(np.float32)
    gf = rgb[:, :, 1].astype(np.float32)
    bf = rgb[:, :, 2].astype(np.float32)
    exg = (2.0 * gf - rf - bf) / (rf + gf + bf + 1.0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tex = np.hypot(gx, gy)

    segs = slic(rgb, n_segments=n_seg, compactness=14.0, sigma=1.0, start_label=0, channel_axis=-1)
    n_lbl = int(segs.max()) + 1
    flat = segs.ravel()
    cnts = np.bincount(flat, minlength=n_lbl).astype(np.float32)
    safe = np.maximum(cnts, 1.0)

    def sp_avg(arr: np.ndarray) -> np.ndarray:
        return np.bincount(flat, weights=arr.ravel(), minlength=n_lbl) / safe

    vm = sp_avg(hsv[:, :, 2])
    v2 = sp_avg(hsv[:, :, 2] ** 2)
    v_std = np.sqrt(np.maximum(v2 - vm ** 2, 0.0))

    ndvi_m = sp_avg(ndvi)
    exg_m = sp_avg(exg)
    sm = sp_avg(hsv[:, :, 1])
    bm = sp_avg(lab[:, :, 2])
    tm = sp_avg(tex)

    feats: List[Dict] = []
    for sid in range(n_lbl):
        feats.append(
            {
                "id": sid,
                "ndvi": float(ndvi_m[sid]),
                "exg": float(exg_m[sid]),
                "sat": float(sm[sid]),
                "val": float(vm[sid]),
                "v_std": float(v_std[sid]),
                "b_lab": float(bm[sid]),
                "tex": float(tm[sid]),
            }
        )
    return segs, feats


def build_masks(segs: np.ndarray, feats: List[Dict], yr: str) -> Dict[str, np.ndarray]:
    n_sp = len(feats)
    soil_lut = np.zeros(n_sp, dtype=np.uint8)
    dry_lut = np.zeros(n_sp, dtype=np.uint8)
    sana_lut = np.zeros(n_sp, dtype=np.uint8)
    grn_cut = 0.60 if yr == "2025" else 0.63

    for ft in feats:
        sid = ft["id"]
        ndvi = ft["ndvi"]
        exg = ft["exg"]
        sat = ft["sat"]
        val = ft["val"]
        v_std = ft["v_std"]
        b_lab = ft["b_lab"]
        tex = ft["tex"]

        if val < 42.0:
            continue

        is_soil = (
            (tex < 95.0 and val > 132.0 and sat < 58.0 and ndvi < 0.45)
            or (ndvi < 0.36 and tex < 110.0 and exg < 0.02)
            or (val > 170.0 and sat < 50.0 and ndvi < 0.48)
        )
        if is_soil:
            soil_lut[sid] = 255
            continue

        is_sana = ndvi > grn_cut or exg > 0.052 or (ndvi > (grn_cut - 0.05) and exg > 0.025)
        if is_sana:
            sana_lut[sid] = 255
            continue

        is_wood = (
            (tex >= 145.0 and v_std >= 38.0 and sat <= 68.0 and 62.0 <= val <= 168.0)
            or (tex >= 130.0 and sat <= 55.0 and b_lab <= 137.0 and 72.0 <= val <= 165.0)
        )
        is_lit = tex >= 82.0 and ndvi >= 0.32 and 52.0 <= val <= 168.0 and sat >= 38.0 and b_lab >= 135.0
        if is_wood or is_lit:
            dry_lut[sid] = 255

    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    return {
        "suelo_desnudo": cv2.morphologyEx(cv2.morphologyEx(soil_lut[segs], cv2.MORPH_CLOSE, kc), cv2.MORPH_OPEN, ko),
        "vegetacion_seca": cv2.morphologyEx(cv2.morphologyEx(dry_lut[segs], cv2.MORPH_CLOSE, kc), cv2.MORPH_OPEN, ko),
        "vegetacion_sana": cv2.morphologyEx(cv2.morphologyEx(sana_lut[segs], cv2.MORPH_CLOSE, kc), cv2.MORPH_OPEN, ko),
    }


def open_holes(mask: np.ndarray, occ: Optional[np.ndarray], min_a: int = 500) -> np.ndarray:
    # abre huecos internos ocupados por clases de mayor prioridad
    if occ is None or not occ.any():
        return mask
    out = mask.copy()
    rows, cols = out.shape
    for step in range(5):
        cnts, hier = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            break
        found = False
        for idx, cnt in enumerate(cnts):
            if hier[0][idx][3] != -1 and cv2.contourArea(cnt) >= min_a:
                hm = np.zeros((rows, cols), dtype=np.uint8)
                cv2.drawContours(hm, [cnt], -1, 255, thickness=-1)
                if not np.any((hm > 0) & (occ > 0)):
                    continue
                found = True
                mom = cv2.moments(cnt)
                if mom["m00"] > 0:
                    cx = int(mom["m10"] / mom["m00"])
                    cy = int(mom["m01"] / mom["m00"])
                else:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx, cy = bx + bw // 2, by + bh // 2
                y_end = 0 if cy < rows // 2 else rows - 1
                cv2.line(out, (cx, cy), (cx, y_end), 0, thickness=5)
        if not found:
            break
    return out


def raster_shapes(shapes: List[Dict], rows: int, cols: int, dil_px: int = 0) -> np.ndarray:
    canvas = np.zeros((rows, cols), dtype=np.uint8)
    for sh in shapes:
        pts = np.array(sh["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)
    if dil_px > 0 and canvas.any():
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_px + 1, 2 * dil_px + 1))
        canvas = cv2.dilate(canvas, ker)
    return canvas


def mask_to_shapes(mask: np.ndarray, lbl: str, occ: Optional[np.ndarray], min_a: int, eps_r: float = 0.003) -> List[Dict]:
    clean = open_holes(mask, occ)
    shapes: List[Dict] = []
    cnts, hier = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_a:
            continue
        approx = cv2.approxPolyDP(cnt, eps_r * cv2.arcLength(cnt, True), True)
        if len(approx) < 4:
            continue
        pts = [[round(float(pt[0][0]), 1), round(float(pt[0][1]), 1)] for pt in approx]
        shapes.append(
            {
                "label": lbl,
                "points": pts,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return shapes


def build_shapes(masks: Dict[str, np.ndarray], rows: int, cols: int) -> Tuple[List[Dict], float]:
    # jerarquia sin solapes: suelo_desnudo > vegetacion_seca > vegetacion_sana
    spec = [("suelo_desnudo", 5500), ("vegetacion_seca", 4500), ("vegetacion_sana", 8000)]
    all_sh: List[Dict] = []
    occ = np.zeros((rows, cols), dtype=np.uint8)
    rasters: Dict[str, np.ndarray] = {}

    for cname, min_a in spec:
        avail = cv2.bitwise_and(masks[cname], cv2.bitwise_not(occ))
        shs = mask_to_shapes(avail, cname, occ, min_a)
        all_sh.extend(shs)
        rasters[cname] = raster_shapes(shs, rows, cols, 0)
        occ = cv2.bitwise_or(occ, raster_shapes(shs, rows, cols, 5))

    ms = rasters["suelo_desnudo"] > 0
    md = rasters["vegetacion_seca"] > 0
    mh = rasters["vegetacion_sana"] > 0
    inter = (ms & md) | (ms & mh) | (md & mh)
    uni = ms | md | mh
    return all_sh, float(inter.sum() / max(int(uni.sum()), 1))


def save_json(path: Path, img_name: str, cols: int, rows: int, shapes: List[Dict]) -> None:
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": rows,
        "imageWidth": cols,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_prev(out_path: Path, bgr: np.ndarray, ndvi: np.ndarray, shapes: List[Dict]) -> None:
    ovl = bgr.copy()
    fil = bgr.copy()
    for sh in shapes:
        col = cls_colors.get(sh["label"], (255, 255, 255))
        pts = np.array(sh["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(fil, [pts], col)
        cv2.polylines(ovl, [pts], True, col, 3)

    bld = cv2.addWeighted(fil, 0.28, ovl, 0.72, 0)
    nu8 = np.clip((ndvi - 0.1) / 0.75 * 255.0, 0, 255).astype(np.uint8)
    ncol = cv2.applyColorMap(nu8, cv2.COLORMAP_JET)
    panel = cv2.resize(np.hstack([bld, ncol]), (1280, 640), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def run_tile(
    rec: Dict,
    proc_dir: Path,
    lme_dir: Path,
    prv_dir: Path,
    n_seg: int,
    do_prev: bool,
) -> Tuple[Counter, bool, float]:
    tid = rec["tile_id"]
    rgb_p = lme_dir / f"{tid}.jpg"
    if not rgb_p.exists():
        rgb_p = proc_dir / rec["rgb_path"]
    idx_p = proc_dir / rec["indices_path"]

    bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
    rows, cols = bgr.shape[:2]

    npz = np.load(str(idx_p))
    ndvi = cv2.resize(npz["ndvi"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    ndre = cv2.resize(npz["ndre"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    gndvi = cv2.resize(npz["gndvi"].astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)

    segs, feats = extract_sp_feats(bgr, ndvi, ndre, gndvi, n_seg)
    masks = build_masks(segs, feats, rec["year"])
    shapes, ov_rat = build_shapes(masks, rows, cols)

    save_json(lme_dir / f"{tid}.json", f"{tid}.jpg", cols, rows, shapes)
    if do_prev:
        save_prev(prv_dir / f"{tid}_preview.jpg", bgr, ndvi, shapes)
    return Counter(sh["label"] for sh in shapes), bool(shapes), ov_rat


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    proc_dir = root / "data" / "processed"
    ann_dir = root / "data" / "annotations"

    parser = argparse.ArgumentParser(description="Genera poligonos en formato LabelMe.")
    parser.add_argument("--seleccion-csv", type=Path, default=ann_dir / "labelme_240_seleccion.csv")
    parser.add_argument("--proc-dir", type=Path, default=proc_dir)
    parser.add_argument("--labelme-dir", type=Path, default=ann_dir / "labelme_240")
    parser.add_argument("--preview-dir", type=Path, default=root / "results" / "previews_labelme")
    parser.add_argument("--n-segments", type=int, default=450)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()

    with open(args.seleccion_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit > 0:
        rows = rows[: args.limit]

    args.labelme_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_previews:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    ov_list: List[float] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(run_tile, rec, args.proc_dir, args.labelme_dir, args.preview_dir, args.n_segments, not args.no_previews)
            for rec in rows
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Generando LabelMe"):
            cnt, has_sh, ov = fut.result()
            counts.update(cnt)
            ov_list.append(ov)

    print(f"Tiles: {len(rows)} | Poligonos: {dict(counts)} | Solape medio: {100.0 * float(np.mean(ov_list)):.3f}%")


if __name__ == "__main__":
    main()
