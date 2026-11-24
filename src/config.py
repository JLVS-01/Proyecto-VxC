from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

data_dir = root_dir / "data"
raw_dir = data_dir / "raw"
proc_dir = data_dir / "processed"
ann_dir = data_dir / "annotations"
labelme_dir = ann_dir / "labelme_240"
sel_csv = ann_dir / "labelme_240_seleccion.csv"
manual_dir = ann_dir / "manual_jorge"
yolo_dir = data_dir / "yolo_dataset"
yolo_yaml = yolo_dir / "dataset.yaml"
split_csv = yolo_dir / "resumen_split.csv"

models_dir = root_dir / "models"
res_dir = root_dir / "results"
metrics_dir = res_dir / "metrics"
figs_dir = res_dir / "figures"
maps_dir = res_dir / "risk_maps"
runs_dir = res_dir / "yolo_runs"

cls_names = ["vegetacion_seca", "vegetacion_sana", "suelo_desnudo"]
cls_to_id = {name: idx for idx, name in enumerate(cls_names)}
