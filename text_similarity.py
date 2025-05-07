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
    """
    评估模型在给定数据集上的性能。

    :param model: 用于评估的预训练模型，通常是 SentenceTransformer 类型。
    :param data_loader: 数据加载器，用于批量加载评估数据。
    :param device: 计算设备，如 'cpu' 或 'cuda'。
    :param threshold: 用于将相似度分数转换为二元预测的阈值，默认为 0.625。
    :return: 包含评估指标（准确率、F1 分数、召回率、综合得分）的字典，
             以及相似度分数和真实标签的数组。
    """
    # 将模型设置为评估模式，关闭 dropout 等训练时使用的特殊层
    model.eval()
    # 用于存储所有批次的第一个句子的嵌入向量
    all_embeddings1 = []
    # 用于存储所有批次的第二个句子的嵌入向量
    all_embeddings2 = []
    # 用于存储所有批次的真实标签
    all_labels = []

    # 不计算梯度，减少内存消耗并加快计算速度
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for batch in data_loader:
            # 从批次中提取第一个句子列表
            sentences1 = batch['sentences1']
            # 从批次中提取第二个句子列表
            sentences2 = batch['sentences2']
            # 从批次中提取真实标签列表
            labels = batch['labels']

            # 对第一个句子列表进行编码，转换为张量并移动到指定设备
            embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
            # 对第二个句子列表进行编码，转换为张量并移动到指定设备
            embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)

            # 将当前批次的第一个句子的嵌入向量添加到列表中
            all_embeddings1.append(embeddings1)
            # 将当前批次的第二个句子的嵌入向量添加到列表中
            all_embeddings2.append(embeddings2)
            # 将当前批次的真实标签添加到列表中
            all_labels.extend(labels)

    # 将所有批次的第一个句子的嵌入向量在第 0 维上拼接成一个大张量
    all_embeddings1 = torch.cat(all_embeddings1, dim=0)
    # 将所有批次的第二个句子的嵌入向量在第 0 维上拼接成一个大张量
    all_embeddings2 = torch.cat(all_embeddings2, dim=0)
    # 将所有真实标签转换为张量并移动到指定设备
    all_labels = torch.tensor(all_labels, dtype=torch.int, device=device)

    # 计算所有句子对的余弦相似度
    similarity = torch.cosine_similarity(all_embeddings1, all_embeddings2, dim=1)
    # 根据阈值将相似度分数转换为二元预测，并转换为 NumPy 数组
    preds = (similarity > threshold).cpu().numpy().astype(int)

    # 计算准确率
    accuracy = accuracy_score(all_labels.cpu().numpy(), preds)
    # 计算 F1 分数
    f1 = f1_score(all_labels.cpu().numpy(), preds)
    # 计算召回率
    recall = recall_score(all_labels.cpu().numpy(), preds)
    # 计算综合得分，F1 分数占 60%，准确率占 20%，召回率占 20%
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
    """
    训练句子相似度模型。

    :param model: 预训练的句子嵌入模型，通常是 SentenceTransformer 类型。
    :param train_data: 训练数据集，包含 'q1', 'q2', 'label' 列的 Pandas DataFrame。
    :param val_data: 验证数据集，包含 'q1', 'q2', 'label' 列的 Pandas DataFrame。
    :param device: 计算设备，如 'cpu' 或 'cuda'。
    :param batch_size: 训练和验证时的批次大小，默认为 64。
    :param epochs: 训练的轮数，默认为 3。
    :param patience: 早停机制的耐心值，即模型性能未提升时继续训练的轮数，默认为 3。
    :param min_delta: 模型性能提升的最小阈值，默认为 1e-4。
    :return: 训练好的模型。
    """
    try:
        # 将模型移动到指定设备（GPU 或 CPU）
        model.to(device)
        print(f"模型是否在GPU上: {next(model.parameters()).device}")

        # 创建训练集数据集对象
        train_dataset = SentencePairDataset(
            train_data['q1'].tolist(),  # 训练集第一个句子列表
            train_data['q2'].tolist(),  # 训练集第二个句子列表
            train_data['label'].tolist()  # 训练集标签列表
        )
        # 创建验证集数据集对象
        val_dataset = SentencePairDataset(
            val_data['q1'].tolist(),  # 验证集第一个句子列表
            val_data['q2'].tolist(),  # 验证集第二个句子列表
            val_data['label'].tolist()  # 验证集标签列表
        )

        # 创建训练集数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,  # 批次大小
            shuffle=True,  # 每个 epoch 打乱数据顺序
            pin_memory=True,  # 将数据提前加载到 GPU 内存，加快训练速度
            num_workers=0,  # Windows 下使用 0 避免多进程问题
            collate_fn=custom_collate_fn  # 自定义数据整理函数
        )
        # 创建验证集数据加载器
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,  # 批次大小
            shuffle=False,  # 验证集不需要打乱数据顺序
            pin_memory=True,  # 将数据提前加载到 GPU 内存，加快验证速度
            num_workers=0,  # Windows 下使用 0 避免多进程问题
            collate_fn=custom_collate_fn  # 自定义数据整理函数
        )

        # 定义训练损失函数，使用余弦相似度损失
        train_loss = losses.CosineSimilarityLoss(model)
        # 定义优化器，使用 Adam 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        # 定义梯度缩放器，用于混合精度训练
        scaler = GradScaler()
        # 定义学习率调度器，每 2 个 epoch 将学习率乘以 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        # 记录最佳得分
        best_score = 0
        # 记录模型性能未提升的轮数
        no_improve_epochs = 0
        
        # 导入时间模块，用于记录每个 epoch 的时间消耗
        import time
        # 开始训练循环，遍历每个 epoch
        for epoch in range(epochs):
            # 记录当前 epoch 的开始时间
            start_time = time.time()
            # 将模型设置为训练模式
            model.train()
            # 初始化总损失
            total_loss = 0
            # 创建进度条，用于显示训练进度
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            # 遍历训练数据加载器中的每个批次
            for batch in train_dataloader:
                # 从批次中提取第一个句子列表
                sentences1 = batch['sentences1']
                # 从批次中提取第二个句子列表
                sentences2 = batch['sentences2']
                # 从批次中提取标签列表，并转换为张量移动到指定设备
                labels = torch.tensor(batch['labels'], dtype=torch.float32, device=device)

                # 清空优化器中的梯度信息
                optimizer.zero_grad()

                # 对第一个句子列表进行分词处理
                features1 = model.tokenize(sentences1)
                # 对第二个句子列表进行分词处理
                features2 = model.tokenize(sentences2)

                # 将分词后的特征移动到指定设备
                features1 = {key: val.to(device) for key, val in features1.items()}
                features2 = {key: val.to(device) for key, val in features2.items()}

                # 计算损失
                loss = train_loss([features1, features2], labels)

                # 使用梯度缩放器进行反向传播
                scaler.scale(loss).backward()
                # 使用梯度缩放器更新模型参数
                scaler.step(optimizer)
                # 更新梯度缩放器的状态
                scaler.update()

                # 累加总损失
                total_loss += loss.item()
                # 更新进度条
                progress_bar.update(1)
                # 更新进度条的后缀信息，显示当前批次的损失
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 关闭进度条
            progress_bar.close()
            # 计算平均训练损失
            avg_loss = total_loss / len(train_dataloader)

            # 在验证集上评估模型
            val_metrics = evaluate(model, val_dataloader, device)
            # 获取当前验证集的综合得分
            current_score = val_metrics['score']

            # 更新学习率
            scheduler.step()
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 记录当前 epoch 的结束时间
            end_time = time.time()
            # 计算当前 epoch 的耗时
            epoch_time = end_time - start_time

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"平均训练损失: {avg_loss:.4f}")
            print(f"验证集评估: Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}")
            print(f"当前得分: {current_score:.4f}")
            print(f"当前学习率: {current_lr:.2e}")
            print(f"本轮耗时: {epoch_time:.2f}秒")

            # 如果当前得分大于最佳得分加上最小阈值，则更新最佳得分和保存模型
            if current_score > best_score + min_delta:
                best_score = current_score
                no_improve_epochs = 0
                # 定义模型保存目录
                save_dir = "saved_models"
                # 创建保存目录，如果目录已存在则不报错
                os.makedirs(save_dir, exist_ok=True)

                # 获取当前目录下的所有模型文件，找到最新编号
                existing_models = [f for f in os.listdir(save_dir) if f.startswith("best_model_") and f.endswith(".pth")]
                if existing_models:
                    latest_model = max(existing_models, key=lambda x: int(x.split('_')[2].split('.')[0]))
                    next_model_number = int(latest_model.split('_')[2].split('.')[0]) + 1
                else:
                    next_model_number = 1

                # 生成模型保存路径
                save_path = os.path.join(save_dir, f"best_model_{next_model_number}.pth")
                # 保存模型
                model.save(save_path)

            else:
                # 模型性能未提升，增加未提升的轮数
                no_improve_epochs += 1
                print(f"模型表现未提升，已经 {no_improve_epochs}/{patience} 个epoch")
                # 如果未提升的轮数达到耐心值，则触发早停机制
                if no_improve_epochs >= patience:
                    print("\nEarly stopping triggered")
                    break

        return model

    except RuntimeError as e:
        print(f"训练过程中出现错误: {e}")
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