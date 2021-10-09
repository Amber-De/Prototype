# import the necessary packages
import logging
import dlib
import time
import numpy as np
import argparse
import cv2
import imutils
from imutils.video import VideoStream
from imutils import face_utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-pt", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# allowing the camera to warm up for 2 seconds
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # reading the frame from the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # creating a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extracting the confidence/probability associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filtering out weak detection by ensuring the 'confidence' is greater than 0.5 in our case
        if confidence < args["confidence"]:
            continue

        # compute the (x,y)-co-ordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

        # drawing the bounding face of the face including the probability
        text = "{:.1f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array

            shape = predictor(gray, rect)
            logging.warning(shape)
            print(shape)
            logging.warning(rect)
            print(rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                pts = np.array([[shape[21]], [shape[39]], [shape[42]], [shape[22]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                isClosed = True

                color = (0, 0, 255)
                thickness = 1

                image = cv2.polylines(frame, [pts], isClosed, color, thickness)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()