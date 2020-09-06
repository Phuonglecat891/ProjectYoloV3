from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import json
import numpy as np
import time
import cv2
import glob
import screeninfo
from time import time

# Load the COCO class labels in which our YOLO model was trained on
labelsPath_Q1 = os.path.join("coco.names")
LABELS_Q1 = open(labelsPath_Q1).read().strip().split("\n")
weightsPath_Q1 = os.path.join("yolov3.weights")
configPath_Q1 = os.path.join("yolov3.cfg")

labelsPath_Q2 = os.path.join("yolo.names")
LABELS_Q2 = open(labelsPath_Q2).read().strip().split("\n")
weightsPath_Q2 = os.path.join("yolov3_custom_train_3800.weights")
configPath_Q2 = os.path.join("yolov3_custom_test.cfg")

# Loading the neural network framework Darknet (YOLO was created based on this framework)
net_Q1 = cv2.dnn.readNetFromDarknet(configPath_Q1,weightsPath_Q1)
net_Q2 = cv2.dnn.readNetFromDarknet(configPath_Q2,weightsPath_Q2)

def predict(image, LABELS, net):
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.2
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)

    np.random.seed(1)
    COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = COLORS[classIDs[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)
    return image

app = Flask(__name__)    

# Lưu đường dẫn đến thư mục uploads và result
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAVE_FOLDER'] = 'static/result'
 
@app.route('/', methods=['GET', 'POST'])
def Q1_update():
    if request.method == 'POST':
        file = request.files['file']
        # Lấy phần mở rộng (VD: .jpg, .png,...)
        extension = os.path.splitext(file.filename)[1]
        # Tạo tên mới cho image
        f_name = str(uuid.uuid4()) + extension

        # Từ file người dùng Upload, lưu tạm vào thư mục
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))

        # Loading image để dự đoán (prediction)
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
        height, width, channels = img.shape
        start = time()
        prediction = predict(img, LABELS_Q1, net_Q1)
        print(time() - start)

        # lưu image mới đã dự đoán vào folder result
        cv2.imwrite(os.path.join(app.config['SAVE_FOLDER'], f_name), prediction)

        # Đưa kết quả vào tab mới
        full_file_name = os.path.join(app.config['SAVE_FOLDER'], f_name)
        return redirect(url_for("Q1_result", img_path=full_file_name))
        
    return render_template('Cau1-upload.html')
    
@app.route('/Q1_result', methods=['GET', 'POST'])
def Q1_result():
    return render_template('Cau1-result.html', f_name = request.args.get('img_path'))

@app.route('/Q2', methods=['GET', 'POST'])
def Q2_update():
    if request.method == 'POST':
        file = request.files['file']
        # Lấy phần mở rộng (VD: .jpg, .png,...)
        extension = os.path.splitext(file.filename)[1]
        # Tạo tên mới cho image
        f_name = str(uuid.uuid4()) + extension

        # Từ file người dùng Upload, lưu tạm vào thư mục
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))

        # Loading image để dự đoán (prediction)
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
        height, width, channels = img.shape
        prediction = predict(img, LABELS_Q2, net_Q2)

        # lưu image mới đã dự đoán vào folder result
        cv2.imwrite(os.path.join(app.config['SAVE_FOLDER'], f_name), prediction)

        # Đưa kết quả vào tab mới
        full_file_name = os.path.join(app.config['SAVE_FOLDER'], f_name)
        return redirect(url_for("Q2_result", img_path=full_file_name))
        
    return render_template('Cau2-upload.html')

@app.route('/Q2_result', methods=['GET', 'POST'])
def Q2_result():
    return render_template('Cau2-result.html', f_name = request.args.get('img_path'))
    
        
if __name__ == '__main__':
    app.run()