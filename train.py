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
    """
    自定义数据集类
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据给定的索引获取数据集中的一个样本。

        :param idx: 样本的索引，用于指定要获取的样本在数据集中的位置。
        :return: 经过转换后的图像张量和对应的标签。
        """
        # 打开指定索引对应的图像文件，并将其转换为RGB格式
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # 如果定义了图像转换操作，则对图像进行转换
        if self.transform:
            image = self.transform(image)
        # 获取指定索引对应的标签
        label = self.labels[idx]
        # 返回转换后的图像和对应的标签
        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """
    训练一个epoch，支持混合精度训练
    """
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
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    progress_bar.close()
    return running_loss / len(train_loader)


def evaluate(model, val_loader, device, verbose=False):
    """''

    评估函数，计算准确率和宏F1分数
    """ ""

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
                batch_f1 = f1_score(
                    batch_labels, batch_preds, average="macro", zero_division=1
                )
                batch_accuracies.append(batch_accuracy)
                batch_f1_scores.append(batch_f1)

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    accuracy_variance = np.var(batch_accuracies) if len(batch_accuracies) > 1 else 0.0
    f1_variance = np.var(batch_f1_scores) if len(batch_f1_scores) > 1 else 0.0

    if verbose and len(batch_accuracies) > 0:
        print("\n批次级别统计:")
        print(f"批次数量: {len(batch_accuracies)}")
        print(f"批次准确率: {[f'{acc:.4f}' for acc in batch_accuracies]}")
        print(f"批次宏F1分数: {[f'{f1:.4f}' for f1 in batch_f1_scores]}")
        print(
            f"批次准确率均值: {np.mean(batch_accuracies):.4f}, 方差: {accuracy_variance:.4f}"
        )
        print(f"批次宏F1均值: {np.mean(batch_f1_scores):.4f}, 方差: {f1_variance:.4f}")

    return accuracy, macro_f1, accuracy_variance, f1_variance


def main():
    # 设置 PyTorch 的随机种子，确保结果可复现
    torch.manual_seed(42)
    # 设置 NumPy 的随机种子，确保结果可复现
    np.random.seed(42)

    # 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义图像预处理的转换操作
    transform = transforms.Compose(
        [
            # 调整图像大小为 224x224
            transforms.Resize((224, 224)),
            # 以 0.5 的概率随机水平翻转图像
            transforms.RandomHorizontalFlip(),
            # 随机旋转图像，旋转角度在 -10 到 10 度之间
            transforms.RandomRotation(10),
            # 随机调整图像的亮度、对比度、饱和度和色相
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            # 将图像转换为张量
            transforms.ToTensor(),
            # 对图像进行归一化处理
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 定义数据集所在的目录
    data_dir = "Data/Data"
    # 用于存储图像文件的路径
    image_paths = []
    # 用于存储图像对应的标签
    labels = []

    # 遍历数据目录下的所有文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            # 从文件名中提取类别 ID 并减 1 以进行索引
            class_id = int(filename.split("_")[0]) - 1
            # 将图像文件的完整路径添加到列表中
            image_paths.append(os.path.join(data_dir, filename))
            # 将图像对应的标签添加到列表中
            labels.append(class_id)

    # 将图像路径列表转换为 NumPy 数组
    image_paths = np.array(image_paths)
    # 将标签列表转换为 NumPy 数组
    labels = np.array(labels)

    # 打印总样本数
    print(f"总样本数: {len(labels)}")
    print("类别分布:")
    # 打印每个类别的样本数量
    for i in range(3):
        print(f"类别 {i+1}: {sum(labels == i)} 个样本")

    # 生成从 0 到样本数量的索引数组
    indices = np.arange(len(image_paths))
    # 随机打乱索引数组
    np.random.shuffle(indices)
    # 根据打乱后的索引重新排列图像路径数组
    image_paths = np.array(image_paths)[indices]
    # 根据打乱后的索引重新排列标签数组
    labels = np.array(labels)[indices]

    # 创建一个字典，将索引映射到对应的图像路径和标签
    data_mapping = {i: (image_paths[i], labels[i]) for i in range(len(image_paths))}

    # 初始化分层 K 折交叉验证对象，将数据集分为 5 折
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
        print(f"\nFold {fold + 1}")

        # 创建训练集数据集对象
        train_dataset = VehicleDataset(
            [data_mapping[i][0] for i in train_idx],
            [data_mapping[i][1] for i in train_idx],
            transform=transform,
        )
        # 创建验证集数据集对象
        val_dataset = VehicleDataset(
            [data_mapping[i][0] for i in val_idx],
            [data_mapping[i][1] for i in val_idx],
            transform=transform,
        )

        # 创建训练集数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        # 创建验证集数据加载器
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

        # 加载预训练的 ResNet-18 模型
        model = models.resnet18(pretrained=True)
        # 修改模型的全连接层，使其输出维度为 3，对应 3 个类别
        model.fc = nn.Linear(model.fc.in_features, 3)
        # 将模型移动到指定设备（GPU 或 CPU）
        model = model.to(device)

        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 定义 Adam 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # 定义学习率调度器，根据验证集得分调整学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )
        # 初始化梯度缩放器，用于混合精度训练
        scaler = GradScaler()

        # 记录当前折的最佳得分
        best_score = 0
        # 早停的耐心值
        patience = 3
        # 记录没有改进的轮数
        no_improve = 0

        # 训练 15 个 epoch
        for epoch in range(15):
            # 记录当前 epoch 的开始时间
            start_time = time.time()
            # 训练一个 epoch 并返回训练损失
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            # 评估模型并返回准确率、宏 F1 分数等指标
            accuracy, macro_f1, _, _ = evaluate(model, val_loader, device)

            # 计算当前得分，准确率占 70%，宏 F1 分数占 30%
            current_score = 0.7 * accuracy + 0.3 * macro_f1
            # 在第 10 个 epoch 后启用动态学习率调度器
            if epoch >= 10:
                scheduler.step(current_score)

            # 记录当前 epoch 的结束时间
            end_time = time.time()
            print(f"\nEpoch {epoch+1}/{15}")
            print(f"训练损失: {train_loss:.4f}")
            print(
                f"验证集: 准确率={accuracy:.4f}, 宏F1={macro_f1:.4f}, 得分={current_score:.4f}"
            )
            print(f"本轮耗时: {end_time - start_time:.2f}秒")

            # 如果当前得分大于最佳得分，则更新最佳得分并保存模型
            if current_score > best_score:
                best_score = current_score
                no_improve = 0
                torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            else:
                # 否则，增加没有改进的轮数
                no_improve += 1
                # 如果没有改进的轮数达到耐心值，则触发早停
                if no_improve >= patience:
                    print(f"在轮次 {epoch+1} 触发早停")
                    break

        # 加载当前折的最佳模型
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        # 对最佳模型进行评估并打印详细信息
        final_accuracy, final_macro_f1, _, _ = evaluate(
            model, val_loader, device, verbose=True
        )
        # 将当前折的最终准确率添加到准确率列表中
        accuracies.append(final_accuracy)
        # 将当前折的最终宏 F1 分数添加到宏 F1 分数列表中
        f1_scores.append(final_macro_f1)

        print(f"\n第 {fold + 1} 折最终结果:")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Macro-F1: {final_macro_f1:.4f}")

    # 计算所有折的平均准确率
    mean_accuracy = np.mean(accuracies)
    # 计算所有折的平均宏 F1 分数
    mean_f1 = np.mean(f1_scores)
    # 计算所有折的准确率方差
    accuracy_variance = np.var(accuracies)
    # 计算所有折的宏 F1 分数方差
    f1_variance = np.var(f1_scores)

    print("\n最终五折交叉验证结果:")
    print(f"Accuracy均值: {mean_accuracy:.4f}, 方差: {accuracy_variance:.4f}")
    print(f"Macro-F1均值: {mean_f1:.4f}, 方差: {f1_variance:.4f}")


if __name__ == "__main__":
    # 在主函数中初始化 accuracies 和 f1_scores
    accuracies = []
    f1_scores = []
    main()
