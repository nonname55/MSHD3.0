import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import logging
from torchvision.models import ResNet18_Weights
import json
import torch.optim as optim

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisasterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # 过滤掉损坏的图片
        self.image_paths = []
        self.labels = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                # 尝试打开图片来验证其完整性
                with Image.open(img_path) as img:
                    img.verify()
                self.image_paths.append(img_path)
                self.labels.append(label)
            except Exception as e:
                logger.warning(f"Skipping corrupted image {img_path}: {str(e)}")
                
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # 返回数据集中的第一张图片作为替代
            # 这不是最优解，但能保证程序继续运行
            return self.__getitem__(0)

class DisasterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DisasterClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_data_paths(root_dir):
    image_paths = []
    labels = []
    label_to_idx = {}
    idx_to_label = {}
    
    for idx, category in enumerate(sorted(os.listdir(root_dir))):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        label_to_idx[category] = idx
        idx_to_label[idx] = category
        
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(idx)
    
    return image_paths, labels, label_to_idx, idx_to_label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 获取数据路径和标签映射
    root_dir = './CDD'
    image_paths, labels, label_to_idx, idx_to_label = get_data_paths(root_dir)
    
    # 保存标签映射，供预测时使用
    with open('label_mapping.json', 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f)
    
    # 初始化模型并训练
    num_classes = len(label_to_idx)
    model = DisasterClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)
    # 创建数据集
    train_dataset = DisasterDataset(train_paths, train_labels, transform)
    val_dataset = DisasterDataset(val_paths, val_labels, transform)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    num_classes = len(label_to_idx)
    model = DisasterClassifier(num_classes)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device=device)

if __name__ == '__main__':
    main()