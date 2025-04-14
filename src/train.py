import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import yaml
from tqdm import tqdm
import numpy as np
from model import SpaceDebrisViT
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load dataset configuration
with open('data/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

class SpaceDebrisDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        # Limit dataset size if specified
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load label
        label = torch.zeros((11, 5))  # 11 classes, 5 values per class (x, y, w, h, confidence)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    label[int(class_id)] = torch.tensor([x, y, w, h, 1.0])
        
        return image, label

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=1, gamma=2)
        
    def forward(self, pred, target):
        # MSE loss for bounding box coordinates
        coord_loss = self.mse(pred[:, :4], target[:, :4])
        
        # Focal loss for confidence scores
        conf_loss = self.focal(pred[:, 4], target[:, 4])
        
        # BCE loss for class predictions
        class_loss = self.bce(pred[:, 5:], target[:, 5:])
        
        # Combine losses with weights
        return coord_loss + 2.0 * conf_loss + 1.5 * class_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Strong data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simple validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SpaceDebrisDataset('data/train', transform=train_transform)
    valid_dataset = SpaceDebrisDataset('data/valid', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model with increased capacity
    model = SpaceDebrisViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=11,
        embed_dim=1024,    # Increased from 768
        depth=12,          # Increased from 8
        num_heads=16,      # Increased from 8
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    ).to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10  # Increased patience
    patience_counter = 0
    best_epoch = 0
    
    print("\nStarting training...")
    print(f"Will stop if validation loss doesn't improve for {patience} epochs")
    print(f"Maximum training epochs: 100\n")
    
    for epoch in range(100):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/100 [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/100 [Valid]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(valid_loader)
        
        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/100:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'models/best_model.pth')
            patience_counter = 0
            print(f'New best model saved! (Validation Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter}/{patience} epochs')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered!')
            print(f'Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
            break
    
    if epoch == 99:  # If we completed all epochs
        print(f'\nTraining completed after all 100 epochs!')
        print(f'Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
    
    print('Training finished!')

if __name__ == '__main__':
    train() 