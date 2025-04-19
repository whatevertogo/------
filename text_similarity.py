import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from datetime import timedelta
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
    return str(timedelta(seconds=int(seconds)))

def check_internet_connection():
    try:
        requests.get("https://huggingface.co", timeout=5)
        print("网络连接正常。")
        return True
    except requests.ConnectionError:
        print("网络连接失败，请检查网络。")
        return False

def evaluate(model, data_loader, device, threshold=0.84):
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
            num_workers=2,
            collate_fn=custom_collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=custom_collate_fn
        )

        train_loss = losses.CosineSimilarityLoss(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        best_score = 0
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            for batch in train_dataloader:
                sentences1 = batch['sentences1']
                sentences2 = batch['sentences2']

                optimizer.zero_grad()

                features1 = model.tokenize(sentences1)
                features2 = model.tokenize(sentences2)

                features1 = {key: val.to(device) for key, val in features1.items()}
                features2 = {key: val.to(device) for key, val in features2.items()}

                loss = train_loss([features1, features2], torch.tensor(batch['labels'], dtype=torch.float32, device=device))

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

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"平均训练损失: {avg_loss:.4f}")
            print(f"验证集评估: Accuracy={val_metrics['accuracy']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}")
            print(f"当前得分: {current_score:.4f}")

            if current_score > best_score + min_delta:
                best_score = current_score
                no_improve_epochs = 0
                print(f"保存最佳模型，得分: {best_score:.4f}")
                model.save('best_model')
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
                       patience=2,
                       min_delta=1e-3)

    print("\n保存最终模型...")
    model.save('chinese_semantic_model_final')
    print("模型已保存为: chinese_semantic_model_final")

if __name__ == "__main__":
    main()