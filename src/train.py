import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# --- CONFIGURATION (Adjusted for your folder structure) ---
BASE_DIR = "../dataset"
CSV_FILE = os.path.join(BASE_DIR, "english.csv")
SAVE_MODEL_PATH = "../models/final_handwriting_model.pth"
SAVE_CLASSES_PATH = "../classes.npy"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# --- STEP 1: DATASET CLASS ---
class HandwritingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        print(f"Pre-loading {len(dataframe)} images into RAM...")
        for idx in range(len(dataframe)):
            img_name = dataframe.iloc[idx, 0]
            label = dataframe.iloc[idx, 2] # label_idx
            
            img_path = os.path.join(BASE_DIR, img_name)
            image = Image.open(img_path).convert('RGB')
            # Resize immediately to save RAM
            image = image.resize((224, 224), Image.LANCZOS)
            
            self.images.append(image)
            self.labels.append(label)
            
            if (idx + 1) % 1000 == 0:
                print(f"Loaded {idx + 1} images...")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training starting on: {device}")

    # 1. Prepare Data
    df = pd.read_csv(CSV_FILE)
    label_encoder = LabelEncoder()
    df['label_idx'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)
    
    # Save classes immediately so inference matches
    np.save(SAVE_CLASSES_PATH, label_encoder.classes_)
    print(f"Classes saved to {SAVE_CLASSES_PATH}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label_idx'], random_state=42
    )

    # 2. Transforms
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAdjustSharpness(2, p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Loaders
    train_ds = HandwritingDataset(train_df, transform=train_transforms)
    val_ds = HandwritingDataset(val_df, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Setup
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)

    # 5. Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"--> Saved new best model to {SAVE_MODEL_PATH}")

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()