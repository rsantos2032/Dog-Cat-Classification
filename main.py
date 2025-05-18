import random
import os
from ImageCNN import ImageCNN
from ImageDataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import shutil
import pandas as pd

NUM_EPOCH = 25
SOURCE_DIR = "data"
TRAIN_DIR = "split/train"
VAL_DIR = "split/val"
TEST_DIR = "split/test"
MODEL_PATH = "model.pth"

def evaluate(model, loader, criterion, device):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    eval_loss /= len(loader)
    accuracy = 100 * correct/total
    return eval_loss, accuracy

def split_data():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(folder, exist_ok=True)
        
    images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg")]
    random.seed(128)
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    for file in train_files:
        shutil.copy(os.path.join(SOURCE_DIR, file), os.path.join(TRAIN_DIR, file))
    for file in val_files:
        shutil.copy(os.path.join(SOURCE_DIR, file), os.path.join(VAL_DIR, file))
    for file in test_files:
        shutil.copy(os.path.join(SOURCE_DIR, file), os.path.join(TEST_DIR, file))
    return train_files, val_files, test_files

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    test_transform = val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    return train_transform, val_transform, test_transform

def get_loaders():
    if (os.path.isdir(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0
        and os.path.isdir(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0
        and os.path.isdir(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0):
        train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".jpg")]
        val_files = [f for f in os.listdir(VAL_DIR) if f.endswith(".jpg")]
        test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")]
    else:
        train_files, val_files, test_files = split_data()
    
    train_transform, val_transform, test_transform = build_transforms()
    
    train_dataset = ImageDataset(train_files, TRAIN_DIR, transform=train_transform)
    val_dataset = ImageDataset(val_files, VAL_DIR, transform=val_transform)
    test_dataset = ImageDataset(test_files, TEST_DIR, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    return train_loader, val_loader, test_loader
        
def main():
    train_loader, val_loader, test_loader = get_loaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch: {epoch+1}/{NUM_EPOCH}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.2f}%")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        
    results = pd.DataFrame({
        "Epoch": list(range(1, NUM_EPOCH + 1)),
        "Train_Losses": train_losses,
        "Val_Losses": val_losses,
        "Val_Accuracies": val_accuracies
    })
    results.to_csv("results.csv", index=False)
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Testing Accuracy: {test_acc:.2f}%")
    
if __name__ == "__main__":
    main()