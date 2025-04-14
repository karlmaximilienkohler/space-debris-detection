from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import os
import numpy as np
from datetime import datetime

def detect_space_debris(image_path, confidence_threshold=0.5):
    # Load YOLO model
    model = YOLO('runs/train/exp/weights/best.pt')
    
    # Load class names
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    classes = data_config['names']
    
    # Run detection
    results = model(image_path, conf=confidence_threshold)[0]
    
    # Load original image for display
    original_image = Image.open(image_path).convert('RGB')
    
    # Create figure for detection visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(original_image)
    
    # Colors for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    # Draw predictions
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, 
            edgecolor=colors[class_id], 
            facecolor='none', 
            alpha=0.7
        )
        plt.gca().add_patch(rect)
        
        # Add label
        label = f'{classes[class_id]}: {confidence:.2f}'
        plt.text(
            x1, y1-5, label, 
            color=colors[class_id], 
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get base name of input imagey
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the detection result
    output_path = f'results/{base_name}_detection_{timestamp}.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Print detection results
    print(f"\nDetection Results for {image_path}:")
    print(f"Number of objects detected: {len(results.boxes)}")
    print("\nDetected objects:")
    for box in results.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        confidence = box.conf[0].cpu().numpy()
        print(f"{classes[class_id]}: {confidence:.2f}")
    print(f"\nResults saved as: {output_path}")

if __name__ == '__main__':
    # Let user input the image path
    image_path = input("Enter the path to your test image (e.g., data/test/images/your_image.jpg): ")
    detect_space_debris(image_path) 