from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_rectangular_box_lines(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    
    if lines is None:
        raise ValueError("No lines detected in the image.")
    
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < abs(y1 - y2):
            vertical_lines.append((x1, y1, x2, y2))
        else:
            horizontal_lines.append((x1, y1, x2, y2))
    
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)
    
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image

def find_black_squares_in_cropped_image(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 250, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_y, max_y = float('inf'), -float('inf')
    
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        min_y = min(min_y, y)
        max_y = max(max_y, y + h)
    
    bounding_boxes.sort(key=lambda box: box[0])
    graph_height = max_y - min_y
    normalized_y_positions = []
    
    for box in bounding_boxes:
        x, y, w, h = box
        if w > 5 and h > 5:
            y_center = y + h // 2
            normalized_y = 1 - ((y_center - min_y) / graph_height)
            tolerance = 0.05
            if not (0.5 - tolerance < normalized_y < 0.5 + tolerance):
                normalized_y_positions.append(normalized_y)
    
    return normalized_y_positions

def check_efficiency(y_positions, efficiency_threshold):
    failed_bins = [i + 1 for i, y in enumerate(y_positions) if y < efficiency_threshold]
    return failed_bins

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            cropped_image = detect_rectangular_box_lines(filepath)
            y_positions = find_black_squares_in_cropped_image(cropped_image)
            failed_bins = check_efficiency(y_positions, 0.8)
            
            return render_template('result.html', failed_bins=failed_bins)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
