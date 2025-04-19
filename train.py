import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

class VehicleDataset(Dataset):
    '''
    自定义数据集类
    '''
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    '''
    训练一个epoch，支持混合精度训练
    '''
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(total=len(train_loader), desc="训练进度")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    progress_bar.close()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, device, verbose=False):
    '''''

    评估函数，计算准确率和宏F1分数
    '''''

    model.eval()
    all_preds = []
    all_labels = []
    batch_accuracies = []
    batch_f1_scores = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            if len(np.unique(batch_labels)) > 1:
                batch_accuracy = accuracy_score(batch_labels, batch_preds)
                batch_f1 = f1_score(batch_labels, batch_preds, average='macro', zero_division=1)
                batch_accuracies.append(batch_accuracy)
                batch_f1_scores.append(batch_f1)
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    accuracy_variance = np.var(batch_accuracies) if len(batch_accuracies) > 1 else 0.0
    f1_variance = np.var(batch_f1_scores) if len(batch_f1_scores) > 1 else 0.0
    
    if verbose and len(batch_accuracies) > 0:
        print("\n批次级别统计:")
        print(f"批次数量: {len(batch_accuracies)}")
        print(f"批次准确率: {[f'{acc:.4f}' for acc in batch_accuracies]}")
        print(f"批次宏F1分数: {[f'{f1:.4f}' for f1 in batch_f1_scores]}")
        print(f"批次准确率均值: {np.mean(batch_accuracies):.4f}, 方差: {accuracy_variance:.4f}")
        print(f"批次宏F1均值: {np.mean(batch_f1_scores):.4f}, 方差: {f1_variance:.4f}")

    return accuracy, macro_f1, accuracy_variance, f1_variance

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_dir = "Data/Data"
    image_paths = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            class_id = int(filename.split('_')[0]) - 1
            image_paths.append(os.path.join(data_dir, filename))
            labels.append(class_id)
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"总样本数: {len(labels)}")
    print("类别分布:")
    for i in range(3):
        print(f"类别 {i+1}: {sum(labels == i)} 个样本")
    
    # 5折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths, labels)):
        print(f"\nFold {fold + 1}")

        train_dataset = VehicleDataset(
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=transform
        )
        val_dataset = VehicleDataset(
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )
        scaler = GradScaler()

        best_score = 0
        patience = 3
        no_improve = 0

        for epoch in range(15):
            start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            accuracy, macro_f1, _, _ = evaluate(model, val_loader, device)

            current_score = 0.7 * accuracy + 0.3 * macro_f1
            if epoch >= 10:  # 在第10个epoch后启用动态学习率调度器
                scheduler.step(current_score)

            end_time = time.time()
            print(f"\nEpoch {epoch+1}/{15}")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证集: 准确率={accuracy:.4f}, 宏F1={macro_f1:.4f}, 得分={current_score:.4f}")
            print(f"本轮耗时: {end_time - start_time:.2f}秒")

            if current_score > best_score:
                best_score = current_score
                no_improve = 0
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"在轮次 {epoch+1} 触发早停")
                    break

        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        final_accuracy, final_macro_f1, _, _ = evaluate(model, val_loader, device, verbose=True)
        accuracies.append(final_accuracy)#将当前准确率添加到列表
        f1_scores.append(final_macro_f1)#将当前验证机宏F1分数添加到列表

        print(f"\n第 {fold + 1} 折最终结果:")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Macro-F1: {final_macro_f1:.4f}")

    mean_accuracy = np.mean(accuracies)#计算平均准确率
    mean_f1 = np.mean(f1_scores)#计算平均宏F1分数
    accuracy_variance = np.var(accuracies)
    f1_variance = np.var(f1_scores)


    print("\n最终五折交叉验证结果:")
    print(f"Accuracy均值: {mean_accuracy:.4f}, 方差: {accuracy_variance:.4f}")
    print(f"Macro-F1均值: {mean_f1:.4f}, 方差: {f1_variance:.4f}")

if __name__ == "__main__":
    main()
