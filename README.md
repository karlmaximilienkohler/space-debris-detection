# Space Debris Detection using YOLOv8

This repository contains the implementation of a space debris detection system using YOLOv8, developed as part of the Designing AI course.

## Main Report

The complete project report can be found in the following file:
- https://docs.google.com/document/d/1axz7BPfr0VJEWtgjHwULlhI2wqHD9cM08ZQ7w8w7cmI/edit?tab=t.0

## Project Structure

```
.
├── src/
│   ├── trainyolo.py       # YOLOv8 training script
│   ├── detect.py          # Detection script
│   └── generate_report.py # Performance report generation
├── data/
│   └── data.yaml          # Dataset configuration
├── Individual Assignment - DAI - Karl M. Kohler.docx  # Complete project report
└── requirements.txt       # Project dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run detection:
```bash
python src/detect.py
```

3. Generate performance report:
```bash
python src/generate_report.py
```

## Model Performance

The YOLOv8 model achieves:
- mAP50: 0.865
- Mean Precision: 0.851
- Mean Recall: 0.767
- Mean F1-Score: 0.792

For detailed analysis and discussion, please refer to the main report document. 
