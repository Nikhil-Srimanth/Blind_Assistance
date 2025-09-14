from flask import Flask, render_template, request, jsonify
import threading
import queue
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import openrouteservice
import time
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

ORS_API_KEY = "5b3ce3597851110001cf6248c755863f0a3b4435bea13cc11dd995a9"

last_detection_results = []
camera_active = False
state = "home"
destination = None
pending_action = None
navigation_steps = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command', methods=['POST'])
def process_command():
    global state, camera_active, destination, pending_action, navigation_steps
    try:
        data = request.json
        command = data.get('command', '').lower()
        logging.info(f"Received command: {command}")

        if state == "home":
            if "object detection" in command:
                state = "object_detection"
                camera_active = True
                pending_action = None
                return jsonify({'message': "Opening object detection page. Say 'detect' to capture or 'go back' to return."})
            elif "go to navigation" in command:
                state = "navigation"
                destination = None
                pending_action = None
                navigation_steps = []
                return jsonify({'message': "Opening navigation page. Please say your destination."})
        
        elif state == "object_detection":
            if "detect" in command:
                pending_action = "detect"
                return jsonify({'message': "Capturing image for object detection."})
            elif "go back" in command:
                state = "home"
                camera_active = False
                last_detection_results = []
                pending_action = None
                return jsonify({'message': "Returning to home page."})
        
        elif state == "navigation":
            if "go back" in command:
                state = "home"
                destination = None
                pending_action = None
                navigation_steps = []
                return jsonify({'message': "Returning to home page."})
            elif destination is None and command not in ["go to navigation", "go back"]:
                destination = command
                calculate_navigation(destination)
                return jsonify({'message': f"Destination set to {destination}. Calculating route now."})
            return jsonify({'message': "Please say your destination or 'go back' to return."})
        
        return jsonify({'message': "Command not recognized."})
    except Exception as e:
        logging.error(f"Command processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_navigation(destination):
    global navigation_steps
    try:
        if not ORS_API_KEY or ORS_API_KEY == "YOUR_ACTUAL_OPENROUTESERVICE_API_KEY_HERE":
            logging.error("ORS API key missing or invalid.")
            return
        
        client = openrouteservice.Client(key=ORS_API_KEY)
        start_coords = [81.0545, 16.3437]
        coords = client.pelias_search(destination)
        if not coords['features']:
            logging.warning(f"Destination '{destination}' not found.")
            return
        end_coords = coords['features'][0]['geometry']['coordinates']
        
        route = client.directions(
            coordinates=[start_coords, end_coords],
            profile='foot-walking',
            format='geojson'
        )
        
        steps = route['features'][0]['properties']['segments'][0]['steps']
        navigation_steps = [
            {
                'instruction': step['instruction'],
                'distance': step.get('distance', 0),
                'coordinates': step.get('way_points', [start_coords])[0]
            } for step in steps
        ]
        logging.info(f"Navigation steps calculated: {navigation_steps}")
    except Exception as e:
        logging.error(f"Navigation error: {str(e)}")

@app.route('/detect', methods=['POST'])
def detect():
    global last_detection_results, pending_action
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        model = YOLO("yolov8n.pt")
        results = model(frame)
        detected_objects = [
            model.names[int(box.cls[0])] for box in results[0].boxes if float(box.conf[0]) > 0.5
        ]
        last_detection_results = detected_objects
        
        pending_action = None
        logging.info(f"Detection completed: {detected_objects}")
        return jsonify({'detected_objects': detected_objects})
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        pending_action = None
        return jsonify({'error': str(e)}), 500

@app.route('/state', methods=['GET'])
def get_state():
    try:
        return jsonify({
            'state': state,
            'camera_active': camera_active,
            'detection_results': last_detection_results,
            'pending_action': pending_action,
            'destination': destination
        })
    except Exception as e:
        logging.error(f"State endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/navigation', methods=['GET'])
def get_navigation():
    try:
        return jsonify({
            'destination': destination,
            'steps': navigation_steps
        })
    except Exception as e:
        logging.error(f"Navigation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)