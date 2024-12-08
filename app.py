from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import io
from io import BytesIO
import pandas as pd
import csv
import re
import base64
import json

# Flask app initialization
app = Flask(__name__)

# Load YOLOv8 model
yolo_model = YOLO("best.pt")

# Load TrOCR model and processor
trocr_model = VisionEncoderDecoderModel.from_pretrained("rayyaa/finetune-trocr")
trocr_processor = TrOCRProcessor.from_pretrained("rayyaa/finetune-trocr")

def perform_ocr(cropped_image):
    # Preproses gambar untuk TrOCR
    pixel_values = trocr_processor(images=cropped_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# Memuat data dari file JSON
with open('regional_code.json', 'r') as file:
    data = json.load(file)

def get_plate_region(plate):
    # Memisahkan kode depan dari plat nomor
    region_code = plate.split()[0].upper()
    
    # Memastikan kode plat valid dan ada dalam data JSON
    if region_code not in data:
        return "Daerah tidak ditemukan"
    
    # Mengambil kode belakang dari plat nomor untuk mencocokkan regex
    last_code = plate.split()[-1].upper()
    
    # Menelusuri regex yang ada dalam data JSON untuk kode wilayah tersebut
    for pattern, region in data[region_code].items():
        if re.match(pattern, last_code):
            return region
    
    return "Daerah tidak ditemukan"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Get the uploaded image
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # YOLO Plate Detection
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
      
    if len(detections) == 0:
        return jsonify({'error': 'No plate detected'}), 404

    # Process all detected plates
    plates_info = []
    for det in detections:
        x_min, y_min, x_max, y_max = det
        plate_image = image.crop((x_min, y_min, x_max, y_max))

        # TrOCR OCR Prediction
        plate_number = perform_ocr(plate_image)

        # Find region
        region = get_plate_region(plate_number)

        # Draw bounding box and plate number on the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("arialbd.ttf", size=25)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 27), plate_number, fill="red", font=font)

        # Convert annotated image to base64
        buffered = BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Append plate info
        plates_info.append({'plate_number': plate_number, 'region': region, 'annotated_image': img_str})

    # Return JSON response with plates information and annotated image
    response = {'plates': plates_info}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
