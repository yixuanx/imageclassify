from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
from utils import load_model, load_imagenet_classes, predict_image

# Initialize Flask application
app = Flask(__name__)

# Preload model and class labels (load only once)
model = load_model()
classes = load_imagenet_classes()

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        # Predict
        results = predict_image(model, image, classes)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=80)  # Run in development mode, modify configuration for production environment