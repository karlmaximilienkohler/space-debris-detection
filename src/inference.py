import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import SpaceDebrisViT
import numpy as np

def load_model(model_path, device):
    model = SpaceDebrisViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    processed_image = transform(image)
    return processed_image, original_size

def postprocess_predictions(predictions, original_size, confidence_threshold=0.5):
    # Convert predictions to numpy array
    predictions = predictions.cpu().numpy()
    
    # Filter by confidence
    mask = predictions[..., 4] > confidence_threshold
    predictions = predictions[mask]
    
    # Convert normalized coordinates to pixel coordinates
    predictions[..., [0, 2]] *= original_size[0] / 224
    predictions[..., [1, 3]] *= original_size[1] / 224
    
    return predictions

def visualize_predictions(image_path, predictions, class_names=['Non-debris', 'Debris']):
    # Load and display the original image
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Draw bounding boxes
    for pred in predictions:
        class_id = pred[0]
        x, y, w, h = pred[1:5]
        confidence = pred[4]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=2,
            edgecolor='red' if class_id == 1 else 'green',
            facecolor='none',
            alpha=0.5
        )
        ax.add_patch(rect)
        
        # Add label
        label = f'{class_names[int(class_id)]}: {confidence:.2f}'
        ax.text(x - w/2, y - h/2 - 5, label, color='white', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model('models/best_model.pth', device)
    
    # Process image
    image_path = 'data/test/sample.jpg'  # Replace with your test image path
    processed_image, original_size = preprocess_image(image_path)
    processed_image = processed_image.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(processed_image)
    
    # Post-process predictions
    predictions = postprocess_predictions(predictions[0], original_size)
    
    # Visualize results
    visualize_predictions(image_path, predictions)

if __name__ == '__main__':
    main() 