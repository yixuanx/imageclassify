from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
# from utils import load_model, load_imagenet_classes, predict_image

import gc
import psutil
# 移除：提前导入load_model等（避免Torch过早加载）

process = psutil.Process()

def get_memory_usage():
    return process.memory_info().rss / (1024 * 1024)

# Initialize Flask application
app = Flask(__name__)

# Preload model and class labels (load only once)
# model = load_model()
# classes = load_imagenet_classes()
# 全局变量：延迟初始化模型（启动时不加载）
model = None
classes = None


# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# 延迟加载模型（第一次请求时才加载）
def lazy_load_model():
    global model, classes
    if model is None:
        print("\n[延迟加载] 开始加载模型和依赖库...")
        # 仅此时导入Torch相关模块
        from utils import load_model, load_imagenet_classes
        load_start = get_memory_usage()
        model = load_model()
        classes = load_imagenet_classes()
        load_end = get_memory_usage()
        print(f"[延迟加载] 模型加载完成，内存增加: {load_end - load_start:.2f} MB "
              f"(当前总内存: {load_end:.2f} MB)")

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    # 第一次请求时触发模型加载
    lazy_load_model();

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Predict
        # results = predict_image(model, image, classes)
        from utils import predict_image  # 延迟导入预测函数
        predict_start = get_memory_usage()
        results = predict_image(model, image, classes)
        predict_end = get_memory_usage()
        print(f"[请求处理] 预测完成，内存变化: {predict_end - predict_start:.2f} MB")

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True, port=80)  # Run in development mode, modify configuration for production environment
    app.run(port=10000)  # Run in development mode, modify configuration for production environment