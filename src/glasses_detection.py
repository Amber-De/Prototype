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

# Declaring the least bright/dark it can be
bright_thres = 0.5
dark_thres = 0.4


# coordinates in a np array function
def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts

    # initialise the list of (x,y) coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop  over the 68 facial landmarks and converting them to
    # a 2-tuple of (x,y)-ordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # returning the list of coordinates
    return coords


def get_centers(img, landmarks):
    EYE_LEFT_OUTER = landmarks[36]
    EYE_LEFT_INNER = landmarks[39]
    EYE_RIGHT_OUTER = landmarks[42]
    EYE_RIGHT_INNER = landmarks[45]

    x = (landmarks[36:40]).T[0]
    y = (landmarks[42:46]).T[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTER[0] + EYE_LEFT_INNER[0]) / 2
    x_right = (EYE_RIGHT_OUTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)  # 画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


def getaligned_face(image, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyesCenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    distance = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / distance
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    alignedFace = cv2.warpAffine(image, M, (desired_w, desired_h))

    return alignedFace


def eyeglass(image):
    image = cv2.GaussianBlur(image, (11, 11), 0)

    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1)
    sobelY = cv2.convertScaleAbs(sobelY)

    edgeness = sobelY

    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure1 * 0.3 + measure2 * 0.7

    if measure > 0.15:
        judge = True
    else:
        judge = False
    return judge


# loop over the frames from the video stream
while True:
    # reading the frame from the video stream
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # creating a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extracting the confidence/probability associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filtering out weak detection by ensuring the 'confidence' is greater than 0.5 in our case
        if confidence < args["confidence"]:
            continue

        # compute the (x,y)-co-ordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")
        faceWidth = endX - startX
        frameWidth = w
        total = (faceWidth / frameWidth) * 100

        # drawing the bounding face of the face including the probability
        text = "{:.1f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        #cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.putText(frame, str(total), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face

            for (x, y) in shape:
                # Light Exposure
                total_pixel = np.size(gray)
                dark_pixel = np.sum(dark_part > 0)
                bright_pixel = np.sum(bright_part > 0)

                if dark_pixel / total_pixel > bright_thres:
                    cv2.putText(frame, "Face is underexposed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                                (255, 255, 255), 1)
                if bright_pixel / total_pixel > dark_thres:
                    cv2.putText(frame, "Face is overexposed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                                (255, 255, 255), 1)

                LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(frame, shape)
                aligned_face = getaligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

                # Glasses Prediction
                judge = eyeglass(aligned_face)
                if judge:
                    cv2.putText(frame, "Please remove Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Glasses detected", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0), 2)

                    # Inner Canthus localisation
                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw them on the image
                    pts = np.array([[shape[21]], [shape[39]], [shape[42]], [shape[22]]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    isClosed = True

                    color = (0, 0, 255)
                    thickness = 1

                    image = cv2.polylines(frame, [pts], isClosed, color, thickness)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
