# Space Debris Detection using YOLOv8

This project implements a space debris detection system using YOLOv8, developed as part of the Designing AI course. The system is designed to detect and track space debris in satellite imagery.

## Main Report

The complete project report can be found in the following file:
- [Individual Assignment - DAI - Karl M. Kohler.docx](Individual%20Assignment%20-%20DAI%20-%20Karl%20M.%20Kohler.docx)

## Project Structure

```
space_debris_detection/
├── src/
│   ├── trainyolo.py       # YOLOv8 training script
│   ├── detect.py          # Detection script
│   └── generate_report.py  # Performance report generation
├── data/
│   └── data.yaml          # Dataset configuration
├── models/
│   └── best_model.pth     # Trained model weights
├── results/
│   └── detection_results/  # Output results from detection
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── Individual Assignment - DAI - Karl M. Kohler.docx  # Complete project report
├── .gitignore              # Files to ignore in the repository
└── LICENSE                 # License information
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run detection**:
   ```bash
   python src/detect.py
   ```

3. **Generate performance report**:
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
