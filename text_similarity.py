import os
from datetime import timedelta
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests

class SentencePairDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return InputExample(texts=[self.sentences1[idx], self.sentences2[idx]], 
                          label=self.labels[idx])

def custom_collate_fn(batch):
    sentences1 = [example.texts[0] for example in batch]
    sentences2 = [example.texts[1] for example in batch]
    labels = [example.label for example in batch]
    return {
        'sentences1': sentences1,
        'sentences2': sentences2,
        'labels': labels
    }

def format_time(seconds):
    '''
    将秒数转换为可读的时间格式
    '''
    return str(timedelta(seconds=int(seconds)))

def check_internet_connection():
    '''
    检查网络连接是否正常
    '''
    try:
        requests.get("https://huggingface.co", timeout=5)
        print("网络连接正常。")
        return True
    except requests.ConnectionError:
        print("网络连接失败，请检查网络。")
        return False

def evaluate(model, data_loader, device, threshold=0.625):
    model.eval()
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            sentences1 = batch['sentences1']
            sentences2 = batch['sentences2']
            labels = batch['labels']

            embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)

            all_embeddings1.append(embeddings1)
            all_embeddings2.append(embeddings2)
            all_labels.extend(labels)

    all_embeddings1 = torch.cat(all_embeddings1, dim=0)
    all_embeddings2 = torch.cat(all_embeddings2, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.int, device=device)

    similarity = torch.cosine_similarity(all_embeddings1, all_embeddings2, dim=1)
    preds = (similarity > threshold).cpu().numpy().astype(int)

    accuracy = accuracy_score(all_labels.cpu().numpy(), preds)
    f1 = f1_score(all_labels.cpu().numpy(), preds)
    recall = recall_score(all_labels.cpu().numpy(), preds)
    score = 0.6 * f1 + 0.2 * accuracy + 0.2 * recall

    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'score': score,
        'similarity': similarity.cpu().numpy(),
        'labels': all_labels.cpu().numpy()
    }

def train_model(model, train_data, val_data, device, batch_size=64, epochs=3, patience=3, min_delta=1e-4):
    try:
        model.to(device)
        print(f"模型是否在GPU上: {next(model.parameters()).device}")

        train_dataset = SentencePairDataset(
            train_data['q1'].tolist(),
            train_data['q2'].tolist(),
            train_data['label'].tolist()
        )
        val_dataset = SentencePairDataset(
            val_data['q1'].tolist(),
            val_data['q2'].tolist(),
            val_data['label'].tolist()
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,  # Windows下使用0避免多进程问题
            collate_fn=custom_collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,  # Windows下使用0避免多进程问题
            collate_fn=custom_collate_fn
        )

        train_loss = losses.CosineSimilarityLoss(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 添加学习率调度器
        
        best_score = 0
        no_improve_epochs = 0
        
        # 增加日志输出，记录每个epoch的时间消耗和学习率变化
        import time
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            total_loss = 0
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            for batch in train_dataloader:
                sentences1 = batch['sentences1']
                sentences2 = batch['sentences2']
                labels = torch.tensor(batch['labels'], dtype=torch.float32, device=device)

                optimizer.zero_grad()

                features1 = model.tokenize(sentences1)
                features2 = model.tokenize(sentences2)

                features1 = {key: val.to(device) for key, val in features1.items()}
                features2 = {key: val.to(device) for key, val in features2.items()}

                loss = train_loss([features1, features2], labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            progress_bar.close()
            avg_loss = total_loss / len(train_dataloader)

            val_metrics = evaluate(model, val_dataloader, device)
            current_score = val_metrics['score']

            scheduler.step()  # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"平均训练损失: {avg_loss:.4f}")
            print(f"验证集评估: Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}")
            print(f"当前得分: {current_score:.4f}")
            print(f"当前学习率: {current_lr:.2e}")
            print(f"本轮耗时: {epoch_time:.2f}秒")

            if current_score > best_score + min_delta:
                best_score = current_score
                no_improve_epochs = 0
                # 修改save_path生成逻辑，使其每次保存模型时自动递增编号
                save_dir = "saved_models"
                os.makedirs(save_dir, exist_ok=True)

                # 获取当前目录下的所有模型文件，找到最新编号
                existing_models = [f for f in os.listdir(save_dir) if f.startswith("best_model_") and f.endswith(".pth")]
                if existing_models:
                    latest_model = max(existing_models, key=lambda x: int(x.split('_')[2].split('.')[0]))
                    next_model_number = int(latest_model.split('_')[2].split('.')[0]) + 1
                else:
                    next_model_number = 1

                save_path = os.path.join(save_dir, f"best_model_{next_model_number}.pth")
                model.save(save_path)

            else:
                no_improve_epochs += 1
                print(f"模型表现未提升，已经 {no_improve_epochs}/{patience} 个epoch")
                if no_improve_epochs >= patience:
                    print("\nEarly stopping triggered")
                    break

        return model

    except RuntimeError as e:
        print(f"预训练模型加载失败: {e}")
        return

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    data = pd.read_csv('csv/train.tsv', sep='\t', header=None, names=['q1', 'q2', 'label'])
    test_data = pd.read_csv('csv/test.csv', sep='\t', header=None, names=['q1', 'q2', 'label'])

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    print("\n训练集类别分布:")
    print(train_data['label'].value_counts())

    if not check_internet_connection():
        return

    print("\n加载预训练模型...")
    try:
        model = SentenceTransformer('shibing624/text2vec-base-chinese')
        print("预训练模型加载成功！")
    except RuntimeError as e:
        print(f"预训练模型加载失败: {e}")
        return

    model = train_model(model, train_data, val_data, device, 
                       batch_size=32,
                       epochs=5,
                       patience=1,
                       min_delta=1e-3)

    # 确保save_path在main函数中定义并传递到测试部分
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 获取当前目录下的所有模型文件，找到最新编号
    existing_models = [f for f in os.listdir(save_dir) if f.startswith("best_model_") and f.endswith(".pth")]
    if existing_models:
        latest_model = max(existing_models, key=lambda x: int(x.split('_')[2].split('.')[0]))
        next_model_number = int(latest_model.split('_')[2].split('.')[0]) + 1
    else:
        next_model_number = 1

    save_path = os.path.join(save_dir, f"best_model_{next_model_number}.pth")
    model.save(save_path)

    # 在main函数中加载最佳模型时使用正确的save_path
    print("\n加载最佳模型进行测试...")
    model = SentenceTransformer(save_path)

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

    test_metrics = evaluate(model, test_dataloader, device)
    print("\n测试集评估结果:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"最终得分: {test_metrics['score']:.4f}")

if __name__ == "__main__":
    main()