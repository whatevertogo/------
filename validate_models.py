import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from train import VehicleDataset, evaluate

def validate_cross_validation_models(data_dir, device):
    """
    使用五折交叉验证生成的模型进行验证，并打印最终结果。
    """
    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据
    image_paths = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            class_id = int(filename.split("_")[0]) - 1
            image_paths.append(os.path.join(data_dir, filename))
            labels.append(class_id)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # 5折交叉验证
    fold_results = []
    for fold in range(5):
        print(f"\n加载 Fold {fold} 的模型并评估...")
        model_path = f"best_model_fold{fold}.pth"

        # 加载模型
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        # 创建验证集加载器
        val_indices = np.arange(len(image_paths)) % 5 == fold
        val_dataset = VehicleDataset(
            image_paths[val_indices],
            labels[val_indices],
            transform=transform,
        )
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

        # 评估模型
        accuracy, macro_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Fold {fold} - 准确率: {accuracy:.4f}, 宏F1: {macro_f1:.4f}")
        fold_results.append((accuracy, macro_f1))

    # 汇总结果
    accuracies = [result[0] for result in fold_results]
    f1_scores = [result[1] for result in fold_results]
    print("\n最终五折交叉验证结果:")
    print(f"Accuracy均值: {np.mean(accuracies):.4f}, 方差: {np.var(accuracies):.4f}")
    print(f"Macro-F1均值: {np.mean(f1_scores):.4f}, 方差: {np.var(f1_scores):.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_dir = "Data/Data"
    validate_cross_validation_models(data_dir, device)
