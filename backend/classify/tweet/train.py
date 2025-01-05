import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import os

# 设置环境变量
os.environ['TRANSFORMERS_HTTP_TIMEOUT'] = '120'

# 加载数据集
df = pd.read_csv('./tweets.csv')

# 数据预处理
def clean_text(text):
    text = str(text).lower().strip()
    return text

# 清理文本数据
df['text'] = df['text'].apply(clean_text)

# 确保没有缺失值
df = df.dropna(subset=['text'])

# 创建标签映射并保存
label_mapping = {
    '0': 'non-disaster',
    '1': 'real-disaster'
}

with open('./tweet_label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

# 数据集划分
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42
)

# 自定义数据集类
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 模型定义
class DisasterClassifier(nn.Module):
    def __init__(self):
        super(DisasterClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)  # 2 classes: disaster or non-disaster

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5):
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估模型
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.4f}')
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_disaster_model.pth')
            print(f'New best model saved with accuracy: {accuracy:.4f}')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 保存tokenizer供后续使用
    tokenizer.save_pretrained('./tweet_tokenizer')

    # 创建数据加载器
    train_dataset = DisasterDataset(train_texts, train_labels, tokenizer)
    test_dataset = DisasterDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = DisasterClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, test_loader, criterion, optimizer, device)
    print("Training completed!")

if __name__ == '__main__':
    main()