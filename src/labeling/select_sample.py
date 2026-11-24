import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    proc_dir = root / "data" / "processed"
    ann_dir = root / "data" / "annotations"

    parser = argparse.ArgumentParser(description="Seleccion estratificada de tiles para etiquetado.")
    parser.add_argument("--metadata-csv", type=Path, default=proc_dir / "metadata_tiles.csv")
    parser.add_argument("--proc-dir", type=Path, default=proc_dir)
    parser.add_argument("--out-dir", type=Path, default=ann_dir / "labelme_240")
    parser.add_argument("--out-csv", type=Path, default=ann_dir / "labelme_240_seleccion.csv")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    with open(args.metadata_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    base_cols = list(rows[0].keys())
    by_yr_wp: Dict[str, Dict[int, List[Dict]]] = {"2025": defaultdict(list), "2026": defaultdict(list)}
    for rec in rows:
        rec["val_p10"] = float(rec["ndvi_p10"])
        rec["val_low"] = float(rec["low_ndvi_ratio"])
        by_yr_wp[rec["year"]][int(rec["waypoint_id"])].append(rec)

    n_yr = args.n // 2
    n_soil = max(2, round(n_yr * 0.22))
    n_grn = max(2, round(n_yr * 0.13))
    n_dry = n_yr - n_soil - n_grn

    picked: List[Dict] = []
    seen = set()

    for yr in ["2025", "2026"]:
        wps = sorted(by_yr_wp[yr].keys())
        all_yr = [rec for wp in wps for rec in by_yr_wp[yr][wp]]

        # tiles con suelo desnudo
        soil_cands = [rec for rec in all_yr if rec["val_low"] >= 0.015]
        soil_cands.sort(key=lambda r: r["val_low"], reverse=True)
        cnt = 0
        for rec in soil_cands:
            if cnt >= n_soil:
                break
            if rec["tile_id"] not in seen:
                seen.add(rec["tile_id"])
                item = dict(rec)
                item["selection_type"] = "suelo_y_seco"
                picked.append(item)
                cnt += 1

        # tiles con vegetacion seca y sana
        max_wp = max(2, (n_dry // max(len(wps), 1)) + 2)
        dry_cands: List[Dict] = []
        for wp in wps:
            sub = [rec for rec in by_yr_wp[yr][wp] if rec["tile_id"] not in seen]
            sub.sort(key=lambda r: r["val_p10"])
            for rnk, rec in enumerate(sub[:max_wp]):
                item = dict(rec)
                item["rank_wp"] = rnk
                item["selection_type"] = "seco_y_sano"
                dry_cands.append(item)

        dry_cands.sort(key=lambda r: (r["rank_wp"], r["val_p10"]))
        cnt = 0
        for rec in dry_cands:
            if cnt >= n_dry:
                break
            if rec["tile_id"] not in seen:
                seen.add(rec["tile_id"])
                picked.append(rec)
                cnt += 1

        # tiles densos en vegetacion sana
        grn_cands = [rec for rec in all_yr if rec["tile_id"] not in seen]
        grn_cands.sort(key=lambda r: r["val_p10"], reverse=True)
        cnt = 0
        for rec in grn_cands:
            if cnt >= n_grn:
                break
            seen.add(rec["tile_id"])
            item = dict(rec)
            item["selection_type"] = "vegetacion_sana_densa"
            picked.append(item)
            cnt += 1

    picked.sort(key=lambda r: (int(r["waypoint_id"]), r["year"], r["tile_id"]))

    if args.clean and args.out_dir.exists():
        for path in args.out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols = base_cols + ["selection_type", "labelme_rgb_path"]
    out_rows = []
    for rec in picked:
        src = args.proc_dir / rec["rgb_path"]
        dst = args.out_dir / src.name
        shutil.copy2(src, dst)
        clean = {col: rec[col] for col in base_cols}
        clean["selection_type"] = rec["selection_type"]
        clean["labelme_rgb_path"] = str(dst)
        out_rows.append(clean)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(out_rows)

    print(f"Tiles seleccionados: {len(out_rows)} | Salida: {args.out_dir}")


if __name__ == "__main__":
    main()
