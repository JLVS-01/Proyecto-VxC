import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Entrena y evalua YOLO11-seg.")
    parser.add_argument("--data", type=Path, default=root / "data" / "yolo_dataset" / "dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--name", type=str, default="yolo11n_seg_train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"

    runs_dir = root / "results" / "yolo_runs"
    mod_dir = root / "models" / "yolo"
    met_dir = root / "results" / "metrics"
    mod_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    tr_res = model.train(
        data=str(args.data),
        project=str(runs_dir),
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        amp=not args.no_amp,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.35,
        translate=0.1,
        hsv_h=0.012,
        hsv_s=0.35,
        hsv_v=0.35,
        mosaic=0.8,
        close_mosaic=15,
        overlap_mask=False,
        mask_ratio=2,
        plots=True,
        save=True,
    )

    exp_dir = Path(tr_res.save_dir)
    best_src = exp_dir / "weights" / "best.pt"
    best_dst = mod_dir / "best.pt"
    shutil.copy2(best_src, best_dst)

    # evaluacion final en split test
    best_mod = YOLO(str(best_dst))
    te_met = best_mod.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name="yolo11n_seg_test",
        exist_ok=True,
        plots=True,
    )

    per_cls = {}
    for cid, cname in best_mod.names.items():
        mp_m, mr_m, map50_m, map95_m = te_met.seg.class_result(cid)
        mp_b, mr_b, map50_b, map95_b = te_met.box.class_result(cid)
        f1_m = (2 * mp_m * mr_m) / (mp_m + mr_m + 1e-9)
        per_cls[cname] = {
            "mask_precision": round(float(mp_m), 4),
            "mask_recall": round(float(mr_m), 4),
            "mask_f1": round(float(f1_m), 4),
            "mask_mAP50": round(float(map50_m), 4),
            "mask_mAP50_95": round(float(map95_m), 4),
            "box_mAP50": round(float(map50_b), 4),
            "box_mAP50_95": round(float(map95_b), 4),
        }

    summary = {
        "model": args.model,
        "best_weights": str(best_dst),
        "imgsz": args.imgsz,
        "test_overall": {
            "mask_precision": round(float(te_met.seg.mp), 4),
            "mask_recall": round(float(te_met.seg.mr), 4),
            "mask_mAP50": round(float(te_met.seg.map50), 4),
            "mask_mAP50_95": round(float(te_met.seg.map), 4),
            "box_precision": round(float(te_met.box.mp), 4),
            "box_recall": round(float(te_met.box.mr), 4),
            "box_mAP50": round(float(te_met.box.map50), 4),
            "box_mAP50_95": round(float(te_met.box.map), 4),
        },
        "test_per_class": per_cls,
    }
    (met_dir / "metricas_test_yolo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"YOLO test mAP50(M)={summary['test_overall']['mask_mAP50']:.4f} | pesos={best_dst}")


if __name__ == "__main__":
    main()
