import argparse
import csv
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm


def read_xmp(path: Path) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(220000)
    match = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", raw, re.S)
    return match.group(0).decode("utf-8", errors="ignore") if match else ""


def get_xmp(xmp: str, key: str) -> Optional[str]:
    match = re.search(rf"(?:drone-dji|Camera):{key}[=>\"]+([^<\"\n]+)", xmp)
    return match.group(1).strip() if match else None


def parse_floats(text: str) -> List[float]:
    return [float(val.strip()) for val in text.split(",") if val.strip()]


def parse_mat(xmp: str, key: str) -> np.ndarray:
    vals = parse_floats(get_xmp(xmp, key))
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def calibrate_band(raw_u16: np.ndarray, xmp: str) -> np.ndarray:
    # calibracion radiometrica con metadatos xmp
    blk = float(get_xmp(xmp, "BlackLevel") or 3200.0)
    gain = float(get_xmp(xmp, "SensorGain") or 1.0)
    exp_us = float(get_xmp(xmp, "ExposureTime") or 1000.0)
    irr = float(get_xmp(xmp, "Irradiance") or 1.0)
    pcam = float(get_xmp(xmp, "SensorGainAdjustment") or 1.0)

    img = np.clip(raw_u16.astype(np.float32) - blk, 0.0, None) / 65535.0

    vig_flag = get_xmp(xmp, "VignettingFlag")
    vig_raw = get_xmp(xmp, "VignettingData")
    if vig_raw and (vig_flag is None or int(vig_flag) == 0):
        coeffs = parse_floats(vig_raw)
        rows, cols = img.shape
        cx = float(get_xmp(xmp, "CalibratedOpticalCenterX") or (cols / 2.0))
        cy = float(get_xmp(xmp, "CalibratedOpticalCenterY") or (rows / 2.0))
        yy, xx = np.indices((rows, cols), dtype=np.float32)
        rad = np.hypot(xx - cx, yy - cy)
        vig = np.ones_like(rad, dtype=np.float32)
        r_pow = rad
        for coeff in coeffs:
            vig += np.float32(coeff) * r_pow
            r_pow = r_pow * rad
        img = img * vig

    denom = gain * (exp_us * 1e-6) * max(irr, 1e-6)
    ref = ((img * pcam) / denom) * 1000.0
    return ref.astype(np.float32)


def dewarp_img(img: np.ndarray, xmp: str) -> np.ndarray:
    # corrige distorsion radial de lente
    flag = get_xmp(xmp, "DewarpFlag")
    raw = get_xmp(xmp, "DewarpData")
    if not raw or (flag is not None and int(flag) != 0):
        return img

    f_cal = float(get_xmp(xmp, "CalibratedFocalLength"))
    cx_cal = float(get_xmp(xmp, "CalibratedOpticalCenterX"))
    cy_cal = float(get_xmp(xmp, "CalibratedOpticalCenterY"))

    parts = raw.split(";")[1]
    fx, fy, dx, dy, k1, k2, p1, p2, k3 = parse_floats(parts)
    k_mat = np.array(
        [[fx, 0.0, cx_cal + dx], [0.0, fy, cy_cal + dy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    k_new = np.array(
        [[f_cal, 0.0, cx_cal], [0.0, f_cal, cy_cal], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cv2.undistort(img, k_mat, dist, None, k_new)


def to_u8(img: np.ndarray) -> np.ndarray:
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, [1.0, 99.0])
    scaled = np.clip((img - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def sift_align(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    sc_ref: float = 0.5,
    sc_mov: float = 0.5,
    fb_mat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    # homografia entre bandas usando sift y ransac
    r_sm = cv2.resize(ref_img, None, fx=sc_ref, fy=sc_ref, interpolation=cv2.INTER_AREA)
    m_sm = cv2.resize(mov_img, None, fx=sc_mov, fy=sc_mov, interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create(nfeatures=4500)
    kp_m, des_m = sift.detectAndCompute(to_u8(m_sm), None)
    kp_r, des_r = sift.detectAndCompute(to_u8(r_sm), None)

    if des_m is None or des_r is None or len(kp_m) < 20 or len(kp_r) < 20:
        return (fb_mat if fb_mat is not None else np.eye(3)), 0, 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des_m, des_r, k=2)
    good = [m1 for m1, m2 in pairs if m1.distance < 0.72 * m2.distance]

    if len(good) < 20:
        return (fb_mat if fb_mat is not None else np.eye(3)), 0, len(good)

    pts_m = np.float32([kp_m[m1.queryIdx].pt for m1 in good]) / sc_mov
    pts_r = np.float32([kp_r[m1.trainIdx].pt for m1 in good]) / sc_ref

    h_mat, mask = cv2.findHomography(pts_m, pts_r, cv2.RANSAC, 5.0)
    if h_mat is None or mask is None or int(mask.sum()) < 20:
        return (fb_mat if fb_mat is not None else np.eye(3)), 0, len(good)
    return h_mat, int(mask.sum()), len(good)


def safe_corr(arr_a: np.ndarray, arr_b: np.ndarray, mask: np.ndarray) -> float:
    va = arr_a[mask].astype(np.float64)
    vb = arr_b[mask].astype(np.float64)
    if va.size < 100 or va.std() < 1e-8 or vb.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])


def pair_captures(raw_dir: Path) -> List[Dict]:
    # empareja capturas 2025 y 2026 por gps cercano
    by_yr: Dict[str, List[Dict]] = {}
    for yr in ["2025", "2026"]:
        yr_dir = raw_dir / yr
        items = []
        for jpg in sorted(yr_dir.glob("*_D.JPG")):
            cap_id = jpg.name.replace("_D.JPG", "")
            xmp = read_xmp(jpg)
            items.append(
                {
                    "year": yr,
                    "capture_id": cap_id,
                    "prefix": str(yr_dir / cap_id),
                    "lat": float(get_xmp(xmp, "GpsLatitude")),
                    "lon": float(get_xmp(xmp, "GpsLongitude")),
                    "rel_alt": float(get_xmp(xmp, "RelativeAltitude") or 0.0),
                }
            )
        by_yr[yr] = items

    ref_25 = by_yr["2025"]
    for idx, item in enumerate(ref_25, start=1):
        item["waypoint_id"] = idx
        item["pair_dist_m"] = 0.0

    for item in by_yr["2026"]:
        best_wp = -1
        best_dist = float("inf")
        for ref in ref_25:
            dlat = (item["lat"] - ref["lat"]) * 111320.0
            dlon = (item["lon"] - ref["lon"]) * 111320.0 * math.cos(math.radians(ref["lat"]))
            dist = math.hypot(dlat, dlon)
            if dist < best_dist:
                best_dist = dist
                best_wp = ref["waypoint_id"]
        item["waypoint_id"] = best_wp
        item["pair_dist_m"] = round(best_dist, 3)

    all_caps = by_yr["2025"] + by_yr["2026"]
    all_caps.sort(key=lambda rec: (rec["waypoint_id"], rec["year"]))
    return all_caps


def tile_starts(start: int, end: int, size: int, stride: int) -> List[int]:
    if end - start < size:
        return []
    coords = list(range(start, end - size + 1, stride))
    last_pos = end - size
    if not coords or coords[-1] < last_pos:
        coords.append(last_pos)
    return coords


def save_check_img(
    out_path: Path,
    rgb_bgr: np.ndarray,
    ndvi_rgb: np.ndarray,
    nir_rgb: np.ndarray,
    r_rgb: np.ndarray,
    g_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    rgb_c = cv2.cvtColor(rgb_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    ndvi_c = ndvi_rgb[y0:y1, x0:x1]
    cir = np.stack(
        [to_u8(nir_rgb[y0:y1, x0:x1]), to_u8(r_rgb[y0:y1, x0:x1]), to_u8(g_rgb[y0:y1, x0:x1])],
        axis=-1,
    )

    tw = 800
    th = int(rgb_c.shape[0] * (tw / rgb_c.shape[1]))
    rgb_s = cv2.resize(rgb_c, (tw, th), interpolation=cv2.INTER_AREA)
    ndvi_s = cv2.resize(ndvi_c, (tw, th), interpolation=cv2.INTER_AREA)
    cir_s = cv2.resize(cir, (tw, th), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=110)
    axes[0].imshow(rgb_s)
    axes[0].set_title("RGB corregido", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(ndvi_s, cmap="RdYlGn", vmin=-0.1, vmax=0.85)
    axes[1].set_title("NDVI calibrado", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    axes[2].imshow(cir_s)
    axes[2].set_title("Falso color CIR", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_capture(
    item: Dict,
    out_dir: Path,
    tile_sz: int = 1024,
    overlap: float = 0.15,
    idx_sz: int = 512,
    save_mb: bool = True,
    save_chk: bool = True,
    low_thr: float = 0.30,
) -> Tuple[Dict, List[Dict]]:
    t0 = time.time()
    yr = item["year"]
    wp = item["waypoint_id"]
    cap_id = item["capture_id"]
    pref = Path(item["prefix"])

    rgb_p = Path(f"{pref}_D.JPG")
    rgb_xmp = read_xmp(rgb_p)
    rgb_dw = dewarp_img(cv2.imread(str(rgb_p), cv2.IMREAD_COLOR), rgb_xmp)
    h_rgb, w_rgb = rgb_dw.shape[:2]
    rgb_g = rgb_dw[:, :, 1].astype(np.float32)

    ms_b: Dict[str, np.ndarray] = {}
    ms_x: Dict[str, str] = {}
    for band in ["G", "R", "RE", "NIR"]:
        bp = Path(f"{pref}_MS_{band}.TIF")
        xmp = read_xmp(bp)
        ms_x[band] = xmp
        cal = calibrate_band(tifffile.imread(str(bp)), xmp)
        ms_b[band] = dewarp_img(cal, xmp)

    h_ms, w_ms = ms_b["G"].shape
    ali: Dict[str, np.ndarray] = {"G": ms_b["G"]}
    val_ms = np.ones((h_ms, w_ms), dtype=np.uint8)
    qa: Dict = {
        "waypoint_id": wp,
        "year": yr,
        "capture_id": cap_id,
        "pair_dist_m": item["pair_dist_m"],
        "gps_lat": round(item["lat"], 7),
        "gps_lon": round(item["lon"], 7),
    }

    ones_ms = np.ones((h_ms, w_ms), dtype=np.uint8)
    for band in ["R", "RE", "NIR"]:
        h_mat, inl, tot = sift_align(ms_b["G"], ms_b[band], sc_ref=0.5, sc_mov=0.5)
        wrp = cv2.warpPerspective(ms_b[band], h_mat, (w_ms, h_ms), flags=cv2.INTER_LINEAR)
        wmk = cv2.warpPerspective(ones_ms, h_mat, (w_ms, h_ms), flags=cv2.INTER_NEAREST)
        ali[band] = wrp
        val_ms &= (wmk > 0).astype(np.uint8)
        inn = val_ms > 0
        inn[:100, :] = False
        inn[-100:, :] = False
        inn[:, :100] = False
        inn[:, -100:] = False
        qa[f"inliers_{band}_G"] = f"{inl}/{tot}"
        qa[f"corr_{band}_G"] = round(safe_corr(ali["G"], wrp, inn), 4)

    if save_mb:
        mb_p = out_dir / "multibanda" / yr / f"wp{wp:02d}_{cap_id}_MS_aligned.tif"
        mb_p.parent.mkdir(parents=True, exist_ok=True)
        stk = np.stack([ali["G"], ali["R"], ali["RE"], ali["NIR"]], axis=0).astype(np.float32)
        tifffile.imwrite(str(mb_p), stk, compression="zlib", photometric="minisblack", metadata={"axes": "CYX"})

    eps = 1e-6
    g_a, r_a, re_a, nir_a = ali["G"], ali["R"], ali["RE"], ali["NIR"]
    ndvi_ms = np.clip((nir_a - r_a) / (nir_a + r_a + eps), -1.0, 1.0).astype(np.float32)
    ndre_ms = np.clip((nir_a - re_a) / (nir_a + re_a + eps), -1.0, 1.0).astype(np.float32)
    gndvi_ms = np.clip((nir_a - g_a) / (nir_a + g_a + eps), -1.0, 1.0).astype(np.float32)

    fb_hd = parse_mat(ms_x["G"], "DewarpHMatrix")
    h_rgb_mat, inl_r, tot_r = sift_align(rgb_g, ali["G"], sc_ref=0.25, sc_mov=0.5, fb_mat=fb_hd)
    qa["inliers_MS_RGB"] = f"{inl_r}/{tot_r}"

    g_rgb = cv2.warpPerspective(g_a, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    r_rgb = cv2.warpPerspective(r_a, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    nir_rgb = cv2.warpPerspective(nir_a, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    ndvi_rgb = cv2.warpPerspective(ndvi_ms, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    ndre_rgb = cv2.warpPerspective(ndre_ms, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    gndvi_rgb = cv2.warpPerspective(gndvi_ms, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR)
    val_rgb = cv2.warpPerspective(val_ms, h_rgb_mat, (w_rgb, h_rgb), flags=cv2.INTER_NEAREST)

    # rectangulo interior valido sin vineteo
    col_ok = np.where(val_ms.mean(axis=0) > 0.92)[0]
    row_ok = np.where(val_ms.mean(axis=1) > 0.92)[0]
    mx0 = int(col_ok[0]) + 12 if col_ok.size else 40
    mx1 = int(col_ok[-1]) - 12 if col_ok.size else w_ms - 40
    my0 = int(row_ok[0]) + 12 if row_ok.size else 40
    my1 = int(row_ok[-1]) - 12 if row_ok.size else h_ms - 40

    crn_ms = np.array([[[mx0, my0]], [[mx1, my0]], [[mx1, my1]], [[mx0, my1]]], dtype=np.float64)
    crn_rgb = cv2.perspectiveTransform(crn_ms, h_rgb_mat).reshape(4, 2)
    x_min = max(int(math.ceil(max(crn_rgb[0, 0], crn_rgb[3, 0]))) + 10, 280)
    x_max = min(int(math.floor(min(crn_rgb[1, 0], crn_rgb[2, 0]))) - 10, w_rgb - 280)
    y_min = max(int(math.ceil(max(crn_rgb[0, 1], crn_rgb[1, 1]))) + 10, 220)
    y_max = min(int(math.floor(min(crn_rgb[2, 1], crn_rgb[3, 1]))) - 10, h_rgb - 220)

    for step in range(40):
        if x_max - x_min <= tile_sz or y_max - y_min <= tile_sz:
            break
        if val_rgb[y_min:y_max, x_min:x_max].all():
            break
        x_min += 4
        x_max -= 4
        y_min += 4
        y_max -= 4

    inn_rgb = np.zeros((h_rgb, w_rgb), dtype=bool)
    inn_rgb[y_min:y_max, x_min:x_max] = True
    qa["corr_MSG_RGB"] = round(safe_corr(rgb_g, g_rgb, inn_rgb), 4)
    qa["valid_box"] = f"{x_min},{y_min},{x_max},{y_max}"
    qa["ndvi_median"] = round(float(np.median(ndvi_rgb[inn_rgb])), 4)
    qa["ndre_median"] = round(float(np.median(ndre_rgb[inn_rgb])), 4)

    if save_chk:
        chk_p = out_dir / "verificacion" / f"wp{wp:02d}_{yr}_{cap_id}_check.jpg"
        save_check_img(
            chk_p, rgb_dw, ndvi_rgb, nir_rgb, r_rgb, g_rgb, (x_min, y_min, x_max, y_max),
            title=f"WP {wp:02d} | {yr} | corr={qa['corr_MSG_RGB']} | NDVI={qa['ndvi_median']}",
        )

    stride = int(round(tile_sz * (1.0 - overlap)))
    xs_pos = tile_starts(x_min, x_max, tile_sz, stride)
    ys_pos = tile_starts(y_min, y_max, tile_sz, stride)

    rgb_dir = out_dir / "tiles_rgb" / yr
    idx_dir = out_dir / "tiles_indices" / yr
    rgb_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    recs: List[Dict] = []
    for r_i, ty in enumerate(ys_pos):
        for c_i, tx in enumerate(xs_pos):
            tid = f"wp{wp:02d}_{yr}_{cap_id}_r{r_i:02d}_c{c_i:02d}"
            t_rgb = rgb_dw[ty : ty + tile_sz, tx : tx + tile_sz]
            t_ndvi = ndvi_rgb[ty : ty + tile_sz, tx : tx + tile_sz]
            t_ndre = ndre_rgb[ty : ty + tile_sz, tx : tx + tile_sz]
            t_gndvi = gndvi_rgb[ty : ty + tile_sz, tx : tx + tile_sz]

            rgb_o = rgb_dir / f"{tid}.jpg"
            cv2.imwrite(str(rgb_o), t_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            n_s = cv2.resize(t_ndvi, (idx_sz, idx_sz), interpolation=cv2.INTER_AREA).astype(np.float16)
            re_s = cv2.resize(t_ndre, (idx_sz, idx_sz), interpolation=cv2.INTER_AREA).astype(np.float16)
            g_s = cv2.resize(t_gndvi, (idx_sz, idx_sz), interpolation=cv2.INTER_AREA).astype(np.float16)

            idx_o = idx_dir / f"{tid}.npz"
            np.savez_compressed(str(idx_o), ndvi=n_s, ndre=re_s, gndvi=g_s)

            nf = n_s.astype(np.float32)
            recs.append(
                {
                    "tile_id": tid,
                    "waypoint_id": wp,
                    "year": yr,
                    "capture_id": cap_id,
                    "gps_lat": round(item["lat"], 7),
                    "gps_lon": round(item["lon"], 7),
                    "row": r_i,
                    "col": c_i,
                    "x0": tx,
                    "y0": ty,
                    "tile_size": tile_sz,
                    "ndvi_mean": round(float(nf.mean()), 4),
                    "ndvi_p10": round(float(np.percentile(nf, 10)), 4),
                    "ndvi_p90": round(float(np.percentile(nf, 90)), 4),
                    "ndre_mean": round(float(re_s.astype(np.float32).mean()), 4),
                    "low_ndvi_ratio": round(float(np.mean(nf < low_thr)), 4),
                    "rgb_path": str(rgb_o.relative_to(out_dir)),
                    "indices_path": str(idx_o.relative_to(out_dir)),
                }
            )

    qa["num_tiles"] = len(recs)
    qa["elapsed_s"] = round(time.time() - t0, 2)
    return qa, recs


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Preprocesamiento multiespectral y tiling.")
    parser.add_argument("--dataset-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--waypoints", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--skip-multibanda", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    caps = pair_captures(args.dataset_dir)
    if args.waypoints:
        wp_set = set(args.waypoints)
        caps = [rec for rec in caps if rec["waypoint_id"] in wp_set]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_rows: List[Dict] = []
    tile_rows: List[Dict] = []

    if args.workers == 1 or len(caps) <= 2:
        for rec in tqdm(caps, desc="Procesando capturas"):
            qa, tiles = process_capture(rec, args.output_dir, args.tile_size, args.overlap, 512, not args.skip_multibanda)
            qa_rows.append(qa)
            tile_rows.extend(tiles)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(process_capture, rec, args.output_dir, args.tile_size, args.overlap, 512, not args.skip_multibanda)
                for rec in caps
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Procesando capturas"):
                qa, tiles = fut.result()
                qa_rows.append(qa)
                tile_rows.extend(tiles)

    qa_rows.sort(key=lambda r: (r["waypoint_id"], r["year"]))
    tile_rows.sort(key=lambda r: (r["waypoint_id"], r["year"], r["row"], r["col"]))

    qa_csv = args.output_dir / "qa_alineacion.csv"
    with open(qa_csv, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(qa_rows)

    tiles_csv = args.output_dir / "metadata_tiles.csv"
    with open(tiles_csv, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(tile_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(tile_rows)

    print(f"Capturas: {len(qa_rows)} | Tiles: {len(tile_rows)} | Tiempo: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
