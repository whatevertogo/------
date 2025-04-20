import os
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from text_similarity import SentencePairDataset, custom_collate_fn, evaluate
import pandas as pd

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试数据
    print("加载测试数据...")
    test_data = pd.read_csv('csv/test.csv', sep='\t', header=None, names=['q1', 'q2', 'label'])

    # 准备测试集加载器
    test_dataset = SentencePairDataset(
        test_data['q1'].tolist(),
        test_data['q2'].tolist(),
        test_data['label'].tolist()
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # 找到最新的模型文件
    save_dir = "saved_models"
    existing_models = [f for f in os.listdir(save_dir) if f.startswith("best_model_") and f.endswith(".pth")]
    if not existing_models:
        print("未找到任何已保存的模型。")
        return

    latest_model = max(existing_models, key=lambda x: int(x.split('_')[2].split('.')[0]))
    model_path = os.path.join(save_dir, latest_model)

    print(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path)

    # 在测试集上评估模型
    print("评估模型...")
    threshold_input = input("请输入评估时使用的阈值（默认值为0.625）: ")
    threshold = float(threshold_input) if threshold_input else 0.625
    test_metrics = evaluate(model, test_dataloader, device, threshold=threshold)

    print("\n测试集评估结果:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"最终得分: {test_metrics['score']:.4f}")

if __name__ == "__main__":
    main()