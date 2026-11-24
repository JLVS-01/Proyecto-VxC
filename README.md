# Deteccion y Segmentacion de Combustible Forestal en Imagenes Aereas Multiespectrales

Proyecto de Vision Artificial para detectar material combustible forestal (`vegetacion_seca`, `vegetacion_sana`, `suelo_desnudo`) usando capturas multitemporales (2025-2026) de dron DJI Mavic 3 Multispectral, comparando KNN, Random Forest, SVM y YOLO11n-seg.

## Estructura

```text
proyecto/
├── README.md
├── requirements.txt
├── labels.txt
├── .gitignore
├── venv/
├── data/
│   ├── raw/                  # Dataset original (2025/ y 2026/)
│   ├── processed/            # Bandas calibradas, tiles RGB e indices (.npz)
│   ├── annotations/
│   │   ├── labelme_240/      # 240 tiles .jpg + .json en formato LabelMe
│   │   └── manual_jorge/     # 45 archivos .json anotados manualmente
│   └── yolo_dataset/         # Dataset YOLO-Seg con split espacial por waypoint
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   └── preprocess_multispectral.py
│   ├── labeling/
│   │   ├── select_sample.py
│   │   ├── auto_labelme.py
│   │   └── labelme_to_yolo.py
│   ├── training/
│   │   ├── common_features.py
│   │   ├── train_knn.py
│   │   ├── train_rf.py
│   │   ├── train_svm.py
│   │   └── train_yolo.py
│   └── evaluation/
│       ├── eval_manual_jorge.py
│       ├── eval_pixel_and_summary.py
│       └── generate_risk_maps.py
├── models/
│   ├── yolo/                 # best.pt
│   ├── knn/
│   ├── random_forest/
│   └── svm/
└── results/
    ├── metrics/              # Tablas CSV y metricas JSON
    ├── figures/              # Graficas 300 DPI
    ├── risk_maps/            # Mapas aereos de riesgo de incendio
    └── yolo_runs/            # Curvas de entrenamiento y validacion YOLO
```
