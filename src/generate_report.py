import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO
import yaml
from datetime import datetime

def generate_report():
    # Load the trained model
    model = YOLO('runs/train/exp/weights/best.pt')
    
    # Run validation on test dataset to get metrics
    results = model.val(data='data/data.yaml', split='test')
    
    # Load dataset configuration
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create report directory
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate timestamp for unique report name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_name = f'space_debris_detection_report_{timestamp}'
    
    # Create report file
    report_path = os.path.join(report_dir, f'{report_name}.txt')
    
    with open(report_path, 'w') as f:
        # Write header
        f.write("Space Debris Detection Model Report\n")
        f.write("==================================\n\n")
        
        # Write model information
        f.write("Model Information:\n")
        f.write("-----------------\n")
        f.write(f"Model Type: YOLOv8n\n")
        f.write(f"Number of Classes: {data_config['nc']}\n")
        f.write(f"Classes: {', '.join(data_config['names'])}\n\n")
        
        # Write validation results
        f.write("Validation Results:\n")
        f.write("-----------------\n")
        
        # Get metrics from validation results
        metrics = results.box
        
        # Write overall metrics (using mean values for precision and recall)
        f.write(f"mAP50: {metrics.map50:.3f}\n")
        f.write(f"mAP50-95: {metrics.map:.3f}\n")
        f.write(f"Mean Precision: {np.mean(metrics.p):.3f}\n")
        f.write(f"Mean Recall: {np.mean(metrics.r):.3f}\n")
        f.write(f"Mean F1-Score: {np.mean(metrics.f1):.3f}\n\n")
        
        # Write class-wise metrics
        f.write("Class-wise Performance:\n")
        f.write("----------------------\n")
        for i, name in enumerate(data_config['names']):
            f.write(f"\n{name}:\n")
            f.write(f"  Precision: {metrics.p[i]:.3f}\n")
            f.write(f"  Recall: {metrics.r[i]:.3f}\n")
            f.write(f"  mAP50: {metrics.map50[i]:.3f}\n")
            f.write(f"  mAP50-95: {metrics.map[i]:.3f}\n")
        
        # Write recommendations
        f.write("\nRecommendations:\n")
        f.write("----------------\n")
        
        # Analyze performance and provide recommendations
        if metrics.map50 < 0.5:
            f.write("- Model performance needs improvement. Consider:\n")
            f.write("  * Increasing training epochs\n")
            f.write("  * Using data augmentation\n")
            f.write("  * Collecting more training data\n")
        else:
            f.write("- Model shows good performance. Consider:\n")
            f.write("  * Testing on more diverse images\n")
            f.write("  * Fine-tuning for specific use cases\n")
        
        # Write conclusion
        f.write("\nConclusion:\n")
        f.write("-----------\n")
        f.write("The model has been successfully evaluated. ")
        f.write(f"Overall mAP50 score of {metrics.map50:.3f} indicates ")
        if metrics.map50 < 0.5:
            f.write("room for improvement in detection accuracy.")
        else:
            f.write("good detection performance.")
    
    print(f"Report generated successfully: {report_path}")
    
    # Generate visualization plots
    plot_dir = os.path.join(report_dir, report_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot class-wise performance
    plt.figure(figsize=(12, 6))
    x = np.arange(len(data_config['names']))
    width = 0.35
    
    plt.bar(x - width/2, metrics.map50, width, label='mAP50')
    plt.bar(x + width/2, metrics.map, width, label='mAP50-95')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Class-wise Performance')
    plt.xticks(x, data_config['names'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_performance.png'))
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(metrics.r, axis=0), np.mean(metrics.p, axis=0))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'precision_recall.png'))
    plt.close()
    
    print(f"Visualization plots saved in: {plot_dir}")

if __name__ == '__main__':
    generate_report() 