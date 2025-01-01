from flask import Flask, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 允许上传的图片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # 打印请求信息，用于调试
    print("收到的请求头:", dict(request.headers))
    print("收到的文件:", request.files)
    print("收到的表单数据:", request.form)
    
    # 检查所有上传的文件
    for field_name, file in request.files.items():
        print(f"字段名: {field_name}, 文件名: {file.filename}")
    
    # 获取第一个上传的文件，无论字段名是什么
    if len(request.files) > 0:
        field_name = list(request.files.keys())[0]
        file = request.files[field_name]
        
        # 检查文件名是否为空
        if file.filename == '':
            return {'error': '没有选择文件'}, 400
        
        # 检查是否是允许的文件类型
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # 保存文件到当前目录
            save_path = os.path.join(os.getcwd(), filename)
            file.save(save_path)
            return {
                'message': '图片上传成功',
                'filename': filename,
                'field_name_used': field_name
            }
        else:
            return {'error': f'不允许的文件类型: {file.filename}'}, 400
    
    return {'error': '没有找到文件', 'available_fields': list(request.files.keys())}, 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)