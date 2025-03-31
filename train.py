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

def main(model_size='tiny', epochs=20, lr=0.001):
    # 启用 cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder("data/train", transform=train_transform)
    val_set = datasets.ImageFolder("data/val", transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    model = get_convnext(model_size=model_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    acc_metric = Accuracy(task="multiclass", num_classes=7).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=7, average='macro').to(device)

    best_val_loss = float('inf')  # 初始化最佳验证损失

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        acc_metric.reset()
        f1_metric.reset()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            # 混合精度前向和反向传播
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            acc_metric.reset()
            f1_metric.reset()
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):
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

        # 保存验证集表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

if __name__ == "__main__":
    main(model_size='large', epochs=20, lr=0.00001)