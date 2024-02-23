from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load YOLOv3 and helmet detection model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = load_model('helmet-nonhelmet_cnn.h5')

# Ensure the "uploads" directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        video_path = os.path.join(uploads_dir, file.filename)
        file.save(video_path)
        process_video(video_path)
        return render_template('result.html', video_path=video_path)

# Rest of your code...


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter('static/output.avi', cv2.VideoWriter_fourcc(*"XVID"), 5, (888, 500))

    ret = True

   # Inside process_video function
    while ret:
        ret, img = cap.read()
        if ret:
            img = imutils.resize(img, height=500)
            height, width = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)


        # Rest of your code...


            confidences = []
            boxes = []
            classIds = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    color = [int(c) for c in (0, 255, 0)]  # Default color
                    # Green for bike, red for number plate
                    if classIds[i] == 0:  # Bike
                        helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                    else:  # Number plate
                        x_h = x - 60
                        y_h = y - 350
                        w_h = w + 100
                        h_h = h + 100
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                        if y_h > 0 and x_h > 0:
                            h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                            c = helmet_or_nohelmet(h_r)
                            cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                            cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)

            writer.write(img)

    writer.release()
    # Create the video writer with MJPG codec
    writer = cv2.VideoWriter('static/output.avi', cv2.VideoWriter_fourcc(*"MJPG"), 5, (888, 500))

    cap.release()

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except:
        pass

if __name__ == '__main__':
    app.run(debug=True)
