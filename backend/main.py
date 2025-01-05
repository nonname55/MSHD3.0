from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image  # 用于读取图片数据

app = Flask(__name__)
CORS(app)

# 允许上传的图片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify', methods=['POST'])
def classify():
    # 打印请求信息，用于调试
    print("收到的请求头:", dict(request.headers))
    print("收到的文件:", request.files)
    
    # 获取第一个上传的文件，无论字段名是什么
    if len(request.files) > 0:
        field_name = list(request.files.keys())[0]
        file = request.files[field_name]
        
        # 检查文件名是否为空
        if file.filename == '':
            return {'error': '没有选择文件'}, 400
        
        # 检查是否是允许的文件类型
        if allowed_file(file.filename):
            # 读取文件内容到内存
            file_bytes = file.read()
            
            try:
                # 使用 PIL 打开图片以验证是否为有效图片
                image = Image.open(io.BytesIO(file_bytes))
                
                # 这里可以添加你的图片分类代码
                # 示例：返回一个模拟的分类结果
                return jsonify({
                    'code': 200,
                    'msg': '图片分类完成',  # 这里可以替换为实际的分类结果
                    'success': True,
                    'data': {
                        'filename': file.filename,
                        'size': len(file_bytes),
                        'format': image.format,
                        'mode': image.mode,
                        'dimensions': image.size
                    }
                })
            
            except Exception as e:
                return {'error': f'图片处理失败: {str(e)}'}, 400
        else:
            return {'error': f'不允许的文件类型: {file.filename}'}, 400
    
    return {'error': '没有找到文件', 'available_fields': list(request.files.keys())}, 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)