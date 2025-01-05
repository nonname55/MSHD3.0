from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image
import io
import json
from torchvision.models import ResNet18_Weights

app = Flask(__name__)
CORS(app)

# 图片分类相关设置
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 加载图片分类的标签映射
try:
    with open('./classify/image/label_mapping.json', 'r') as f:
        image_label_mapping = json.load(f)
        idx_to_label = image_label_mapping['idx_to_label']
except FileNotFoundError:
    raise FileNotFoundError("图片分类标签映射文件不存在，请检查路径")

# 加载推文分类的标签映射
try:
    with open('./classify/tweet/tweet_label_mapping.json', 'r') as f:
        tweet_label_mapping = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("推文分类标签映射文件不存在，请检查路径")

# 图片分类模型定义
class ImageDisasterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageDisasterClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 推文分类模型定义
class TweetDisasterClassifier(nn.Module):
    def __init__(self):
        super(TweetDisasterClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)

# 加载模型
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图片分类模型
    try:
        image_model = ImageDisasterClassifier(len(idx_to_label))
        image_model.load_state_dict(torch.load('./classify/image/best_model.pth', map_location=device))
        image_model.eval()
        image_model.to(device)
    except Exception as e:
        raise RuntimeError(f"图片分类模型加载失败: {str(e)}")

    # 加载推文分类模型
    try:
        tweet_model = TweetDisasterClassifier()
        tweet_model.load_state_dict(torch.load('./classify/tweet/best_disaster_model.pth', map_location=device))
        tweet_model.eval()
        tweet_model.to(device)
    except Exception as e:
        raise RuntimeError(f"推文分类模型加载失败: {str(e)}")

    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('./classify/tweet/tweet_tokenizer')
    except Exception as e:
        raise RuntimeError(f"推文分类Tokenizer加载失败: {str(e)}")

    return image_model, tweet_model, tokenizer, device

image_model, tweet_model, tokenizer, device = load_models()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify/image', methods=['POST'])
def classify_image():
    if len(request.files) == 0:
        return jsonify({'error': '没有找到文件'}), 400

    file = list(request.files.values())[0]
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不允许的文件类型: {file.filename}'}), 400

    try:
        # 读取和预处理图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            outputs = image_model(image_tensor)
            _, predicted = outputs.max(1)
            predicted_idx = predicted.item()
            predicted_label = idx_to_label[str(predicted_idx)]

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
        return jsonify({'error': f'预测过程出错: {str(e)}'}), 500

@app.route('/classify/tweet', methods=['POST'])
def classify_tweet():
    if not request.is_json:
        return jsonify({'error': '需要JSON格式的数据'}), 400

    data = request.get_json()
    tweet_text = data.get('text')
    location = data.get('location', '')

    if not tweet_text:
        return jsonify({'error': '没有提供推文内容'}), 400

    try:
        # 预处理推文
        encodings = tokenizer(
            tweet_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # 获取预测结果
        with torch.no_grad():
            outputs = tweet_model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # 返回预测的 target 字段（通常是 0 或 1）
        result = {
            'code': 200,
            'success': True,
            'data': {
                'target': prediction,  # 预测目标类别 0 或 1
                'confidence': f'{confidence:.2%}',
                'location': location if prediction == 1 else None,
                'text': tweet_text
            }
        }

        # 如果是真实灾害且有位置信息，添加额外信息
        if prediction == 1 and location:
            result['data']['message'] = (
                f'检测到真实灾害！位置信息：{location}。'
                f'此信息可用于多源灾害信息获取。'
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'预测过程出错: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
