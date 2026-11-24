import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


cls_to_id = {
    "vegetacion_seca": 0,
    "vegetacion_sana": 1,
    "suelo_desnudo": 2,
}


def parse_wp_yr(tid: str) -> Tuple[int, str]:
    match = re.match(r"^wp(\d+)_(\d{4})_", tid)
    if not match:
        raise ValueError(f"Nombre invalido: {tid}")
    return int(match.group(1)), match.group(2)


def json_to_yolo(json_p: Path) -> Tuple[List[str], Counter]:
    data = json.loads(json_p.read_text(encoding="utf-8"))
    cols = float(data["imageWidth"])
    rows = float(data["imageHeight"])
    lines: List[str] = []
    counts: Counter = Counter()

    for sh in data.get("shapes", []):
        lbl = sh.get("label", "").strip()
        if lbl not in cls_to_id:
            continue
        pts = sh.get("points", [])
        if len(pts) < 3:
            continue
        coords: List[str] = []
        for px, py in pts:
            coords.append(f"{min(max(float(px) / cols, 0.0), 1.0):.6f}")
            coords.append(f"{min(max(float(py) / rows, 0.0), 1.0):.6f}")
        lines.append(f"{cls_to_id[lbl]} " + " ".join(coords))
        counts[lbl] += 1
    return lines, counts


def split_wps(wps: List[int], tr_rat: float = 0.70, va_rat: float = 0.16, seed: int = 42) -> Dict[int, str]:
    # particion espacial por waypoint completo
    uniq = sorted(set(wps))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    tot = len(uniq)
    n_tr = max(1, int(round(tot * tr_rat)))
    n_va = max(1, int(round(tot * va_rat)))
    if n_tr + n_va >= tot:
        n_va = max(1, (tot - n_tr) // 2)

    smap: Dict[int, str] = {}
    for idx, wp in enumerate(uniq):
        if idx < n_tr:
            smap[wp] = "train"
        elif idx < n_tr + n_va:
            smap[wp] = "val"
        else:
            smap[wp] = "test"
    return smap


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Convierte LabelMe a dataset YOLO-Seg con split espacial.")
    parser.add_argument("--labelme-dir", type=Path, default=root / "data" / "annotations" / "labelme_240")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "yolo_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    args = parser.parse_args()

    items = []
    for jf in sorted(args.labelme_dir.glob("*.json")):
        tid = jf.stem
        img_p = args.labelme_dir / f"{tid}.jpg"
        if not img_p.exists():
            continue
        wp, yr = parse_wp_yr(tid)
        items.append({"tid": tid, "wp": wp, "yr": yr, "json": jf, "img": img_p})

    smap = split_wps([it["wp"] for it in items], args.train_ratio, args.val_ratio, args.seed)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    for sp in ["train", "val", "test"]:
        (args.out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    sp_imgs = Counter()
    sp_cls: Dict[str, Counter] = defaultdict(Counter)
    sp_wps: Dict[str, set] = defaultdict(set)
    rows = []

    for it in items:
        sp = smap[it["wp"]]
        lines, cnt = json_to_yolo(it["json"])
        shutil.copy2(it["img"], args.out_dir / "images" / sp / f"{it['tid']}.jpg")
        (args.out_dir / "labels" / sp / f"{it['tid']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        sp_imgs[sp] += 1
        sp_wps[sp].add(it["wp"])
        sp_cls[sp].update(cnt)
        rows.append(
            {
                "tile_id": it["tid"],
                "split": sp,
                "waypoint_id": it["wp"],
                "year": it["yr"],
                "n_vegetacion_seca": cnt["vegetacion_seca"],
                "n_vegetacion_sana": cnt["vegetacion_sana"],
                "n_suelo_desnudo": cnt["suelo_desnudo"],
                "total_poligonos": len(lines),
            }
        )

    assert sp_wps["train"].isdisjoint(sp_wps["val"])
    assert sp_wps["train"].isdisjoint(sp_wps["test"])
    assert sp_wps["val"].isdisjoint(sp_wps["test"])

    yaml_txt = (
        f"path: {args.out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: vegetacion_seca\n"
        f"  1: vegetacion_sana\n"
        f"  2: suelo_desnudo\n"
    )
    (args.out_dir / "dataset.yaml").write_text(yaml_txt, encoding="utf-8")

    csv_p = args.out_dir / "resumen_split.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    for sp in ["train", "val", "test"]:
        print(f"{sp}: {sp_imgs[sp]} imgs | {dict(sp_cls[sp])}")


if __name__ == "__main__":
    main()
