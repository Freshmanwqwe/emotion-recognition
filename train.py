import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

def get_convnext(model_size='tiny', num_classes=7):
    if model_size == 'tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights)
    elif model_size == 'small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights)
    elif model_size == 'base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights)
    else:
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def main(model_size='large', epochs=10, lr=0.01, grad_accum_steps=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuning

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder("datasets/emotion/train", transform=train_transform)
    val_set = datasets.ImageFolder("datasets/emotion/val", transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, pin_memory=True)

    model = get_convnext(model_size=model_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training

    acc_metric = Accuracy(task="multiclass", num_classes=7).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=7, average='macro').to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        acc_metric.reset()
        f1_metric.reset()

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):  # Mixed precision context
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)

        epoch_loss = running_loss / len(train_set)
        epoch_acc = acc_metric.compute().item()
        epoch_f1 = f1_metric.compute().item()

        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            acc_metric.reset()
            f1_metric.reset()
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):  # Mixed precision context
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                acc_metric.update(preds, labels)
                f1_metric.update(preds, labels)
            val_loss /= len(val_set)
            val_acc = acc_metric.compute().item()
            val_f1 = f1_metric.compute().item()
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}\n")
        if epoch % 5 == 4:
            model_save_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main(model_size='large', epochs=20, lr=0.0001, grad_accum_steps=1)