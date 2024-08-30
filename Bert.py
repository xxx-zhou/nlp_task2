import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 数据集加载和预处理
class RottenTomatoesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(csv_file):
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    # 提取句子和标签
    sentences = data['review'].tolist()
    labels = data['sentiment'].tolist()
    return sentences, labels

# 调用函数
csv_file = 'train.tsv'  # 替换为你的文件路径
sentences, labels = load_dataset(csv_file)


# 数据集划分
train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2)

# 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # 可以根据数据集进行调整

# 创建数据集
train_dataset = RottenTomatoesDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = RottenTomatoesDataset(val_texts, val_labels, tokenizer, max_length)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 模型初始化
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_loader) * 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练函数
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

# 评估函数
def evaluate_epoch(model, data_loader):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_accuracy += accuracy_score(batch['labels'], predictions)
    return total_accuracy / len(data_loader)

# 训练模型
for epoch in range(4):  
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_accuracy = evaluate_epoch(model, val_loader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')