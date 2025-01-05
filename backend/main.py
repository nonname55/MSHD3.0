from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
from torchvision.models import ResNet18_Weights

app = Flask(__name__)
CORS(app)

# 允许上传的图片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 加载标签映射
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
    idx_to_label = label_mapping['idx_to_label']

# 定义模型类
class DisasterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DisasterClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 加载模型
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterClassifier(len(idx_to_label))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    model.to(device)
    return model

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 加载模型
model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify', methods=['POST'])
def classify():
    if len(request.files) == 0:
        return {'error': '没有找到文件'}, 400

    file = list(request.files.values())[0]
    
    if file.filename == '':
        return {'error': '没有选择文件'}, 400
        
    if not allowed_file(file.filename):
        return {'error': f'不允许的文件类型: {file.filename}'}, 400

    try:
        # 读取图片并预处理
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            predicted_idx = predicted.item()
            predicted_label = idx_to_label[str(predicted_idx)]  # JSON中的键是字符串
            
            # 获取预测概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx].item()

        return jsonify({
            'code': 200,
            'success': True,
            'msg': f'预测结果: {predicted_label}',
            'data': {
                'predicted_class': predicted_label,
                'confidence': f'{confidence:.2%}',
                'filename': file.filename
            }
        })

    except Exception as e:
        return {'error': f'预测过程出错: {str(e)}'}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)