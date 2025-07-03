import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('code_language_model.keras')

class_labels = ['Python', 'HTML', 'JavaScript']

app = Flask(__name__, static_folder='static')


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error="No selected file")

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  

        preds = model.predict(x)
        class_idx = np.argmax(preds[0])
        predicted_class = class_labels[class_idx]

        return render_template('upload.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
