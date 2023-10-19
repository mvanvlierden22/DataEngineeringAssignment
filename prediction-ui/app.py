from flask import Flask, render_template, request
import requests
import cv2
import base64
import numpy as np
import json

# Flask constructor
app = Flask(__name__)

# Update to correct endpoint
PREDICTION_API_ENDPOINT = 'prediction_api_endpoint'

@app.route('/')
def index():
    return render_template('input_form_page.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get input data from the form
    # input_data = request.form.get('inputData')

    # Check if a file is included in the request
    if 'imageFile' in request.files:
        image_file = request.files['imageFile']

        # Process the image file
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Make a POST request to the prediction API
        files = {'file': (image_file.filename, image_data, 'image/jpeg')}
        response = requests.post(PREDICTION_API_ENDPOINT, files=files)

        # Process the API response
        if response.status_code == 200:
            # Extract bounding box information from the response
            predictions = response.json()

            # Display the image with bounding boxes
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            image_with_boxes = image.copy()

            for prediction in predictions:
                box = prediction['box']
                class_name = prediction['name']
                confidence = prediction['confidence']

                # Convert box coordinates to integers
                x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

                # Draw bounding box and label on the image
                color = (0, 255, 0)  # Green
                label = f'Class: {class_name}, Confidence: {confidence:.2f}'
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert the image to RGB (OpenCV uses BGR by default)
            image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB).astype(np.uint8)

            # Encode the image with bounding boxes to base64 for display in the UI
            image_with_boxes_base64 = base64.b64encode(cv2.imencode('.png', image_with_boxes_rgb)[1]).decode('utf-8')

            # Render the result template with the image and bounding box predictions
            return render_template('response_page.html', image_base64=image_base64, image_with_boxes_base64=image_with_boxes_base64)
        else:
            error_message = f"Error from YOLOv8 API: {response.status_code}"
            return render_template('error.html', error_message=error_message)

    # Handle case where no image is uploaded
    return render_template('error.html', error_message="No image uploaded.")

if __name__ == '__main__':
    app.run(debug=True)