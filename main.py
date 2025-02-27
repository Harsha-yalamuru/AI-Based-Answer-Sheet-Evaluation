from flask import Flask, render_template, request
import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from pdf2image import convert_from_path

# If needed, set the path to Tesseract manually:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust path if necessary

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained CNN model for digit recognition from the specified path
model = tf.keras.models.load_model("C:/Users/harsh/Desktop/mini proj 6thsem/handwriting_model.h5")  # Updated model path

def extract_text(image_path):
    # If the file is a PDF, convert to image first
    if image_path.endswith('.pdf'):
        pages = convert_from_path(image_path, dpi=600)  # Higher DPI for better quality
        image = np.array(pages[0])  # Convert first page to image
    else:
        image = cv2.imread(image_path)

    # Preprocess image: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to create a binary image
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Alternatively, use a basic thresholding technique
    # _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use pytesseract with a custom config for better results
    custom_config = r'--oem 3 --psm 3'  # Try different PSM modes like 3 or 11
    text = pytesseract.image_to_string(gray, config=custom_config)

    return text

def predict_digit(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28)) / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(image))
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            extracted_text = extract_text(file_path)
            return render_template('result.html', text=extracted_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
