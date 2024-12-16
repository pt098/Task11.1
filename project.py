import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import serial
import time
import requests
from flask import Flask, request
from threading import Thread

app = Flask(__name__)

def connect_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        print("Serial connection established.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

ser = connect_serial()

def enable_gps():
    if ser:
        ser.write(b'AT+CGNSSPWR=1\r')
        time.sleep(1)
        print("GPS enabled.")

def get_gps_location(retries=2):
    global ser
    if not ser:
        print("Serial connection is not available.")
        return None, None
    for attempt in range(retries):
        try:
            ser.write(b'AT+CGNSSINFO\r')
            time.sleep(2)
            response = ser.readlines()
            for line in response:
                line_decoded = line.decode().strip()
                print(f"Received line: {line_decoded}")
                if '+CGNSSINFO:' in line_decoded:
                    gps_data = line_decoded.split(",")
                    try:
                        latitude = gps_data[5]
                        lat_dir = gps_data[6]
                        longitude = gps_data[7]
                        lon_dir = gps_data[8]
                        lat = convert_to_degrees(latitude, lat_dir)
                        lon = convert_to_degrees(longitude, lon_dir)
                        print(f"GPS Location: Latitude = {lat}, Longitude = {lon}")
                        return lat, lon
                    except IndexError:
                        print("Failed to parse GPS data.")
        except serial.SerialException as e:
            print(f"Error reading GPS data: {e}")
            ser.close()
            ser = connect_serial()
    print("GPS data not available after retries.")
    return None, None

def convert_to_degrees(coordinate, direction):
    if coordinate and direction:
        degrees = float(coordinate[:2])
        minutes = float(coordinate[2:])
        decimal_degrees = degrees + (minutes / 60)
        return -decimal_degrees if direction in ['S', 'W'] else decimal_degrees
    return None

def send_sms(phone_number, message):
    if ser:
        try:
            ser.write(b'AT+CMGF=1\r')
            time.sleep(1)
            ser.write(f'AT+CMGS="{phone_number}"\r'.encode())
            time.sleep(1)
            ser.write(f'{message}\x1A'.encode())
            time.sleep(1)
            print("SMS sent!")
        except Exception as e:
            print(f"Error sending SMS: {e}")

def make_call(phone_number):
    if ser:
        try:
            ser.write(f'ATD{phone_number};\r'.encode())
            time.sleep(1)
            print(f"Calling {phone_number}...")
        except Exception as e:
            print(f"Error making call: {e}")

@app.route('/alert', methods=['POST'])
def alert():
    alert_type = request.form.get("type")
    print(f"Alert Received: {alert_type}")
    enable_gps()
    lat, lon = get_gps_location()
    if alert_type == "tsunami":
        message = "Tsunami alert triggered on Raspberry Pi!"
    elif alert_type == "earthquake":
        message = "Earthquake alert on Raspberry Pi may cause Tsunami!"
    if lat is not None and lon is not None:
        message += f"\nLocation: Latitude = {lat}, Longitude = {lon}"
    else:
        message += "\nLocation: GPS data not available."
    send_sms("+918700260958", message)
    make_call("+918700260958")
    if ser:
        ser.write(b'ACTIVATE_BUZZER_LED\n')  # Send command to Arduino
    return "Alert Processed"

def start_flask():
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

def trigger_alert(alert_type):
    try:
        response = requests.post("http://127.0.0.1:5000/alert", data={"type": alert_type})  # Include alert type in the request
        if response.status_code == 200:
            print("Alert Processed and SMS Sent!")
        else:
            print("Failed to process alert.")
    except Exception as e:
        print(f"Failed to send alert: {e}")

def detect_tsunami():
    print("Loading model...")
    model_path = 'tsunami_detection_model.keras'
    model = load_model(model_path)
    print("Model loaded.")

    def preprocess_image(image):
        img = cv2.resize(image, (256, 256))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    print("Initializing camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
    picam2.start()
    print("Camera started.")

    while True:
        print("Capturing frame...")
        frame = picam2.capture_array()
        print("Frame captured.")

        print("Preprocessing frame...")
        processed_frame = preprocess_image(frame)
        print("Frame preprocessed.")

        print("Predicting frame...")
        prediction = model.predict(processed_frame, verbose=1)
        print(f"Prediction: {prediction}")

        if prediction[0][0] > 0.7:  # Adjusted threshold to 0.7
            print("Tsunami Detected!")
            trigger_alert("tsunami")
            if ser:
                ser.write(b'ACTIVATE_BUZZER_LED\n')  # Consistent command
                print("Alert sent to Arduino.")

        label_text = "Tsunami Wave Detected" if prediction[0][0] > 0.7 else "Normal Wave"
        frame = cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Tsunami Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


def listen_for_arduino():
    while True:
        if ser and ser.in_waiting > 0:
            alert_msg = ser.readline().decode().strip()
            print(f"Received from Arduino: {alert_msg}")
            if alert_msg == "TSUNAMI_ALERT":
                print("Tsunami alert signal received from Arduino!")
                trigger_alert("tsunami")
            elif alert_msg == "EARTHQUAKE_ALERT":
                print("Earthquake alert signal received from Arduino!")
                trigger_alert("earthquake")

if __name__ == "__main__":
    print("Starting Flask and detection threads...")
    flask_thread = Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    arduino_thread = Thread(target=listen_for_arduino)
    arduino_thread.daemon = True
    arduino_thread.start()

    detect_tsunami()  # Start the tsunami detection process
