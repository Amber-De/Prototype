# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import dlib
import os
import numpy as np
import argparse
import cv2
from imutils import face_utils
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to image")
ap.add_argument("-ii", "--image2", required=False,
                help="path to image")
ap.add_argument("-f", "--face", type=str,
                default="/Users/amberdebono/PycharmProjects/Prototype/src",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="/Users/amberdebono/PycharmProjects/Prototype/src/mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
# It is noted that with 0.5 confidence level the glasses and inner canthus are not detected, whilst with 0.3 they are
# detected
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Declaring the least bright/dark it can be
bright_thres = 0.5
dark_thres = 0.4


def detect_and_predict_mask(frame, net, model):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


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


def perspective_transformation(optical):
    # height, width
    rows, cols, ch = optical.shape

    # Optical Image Points
    pts1 = np.float32([[893, 275], [1537, 257], [925, 1173], [1525, 1125]])
    # Thermal Image Points
    pts2 = np.float32([[527, 79], [1023, 74], [527, 595], [1002, 563]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(optical, M, (cols, rows))

    return dst


def inner_canthus_coords_thermal(image1, shape):

    topleft = shape[21]
    bottomleft = shape[39]
    topright = shape[22]
    bottomright = shape[42]

    topleftbar = (1233, 97)
    bottomleftbar = (1233, 616)
    toprightbar = (1277, 97)
    bottomrightbar = (1277, 616)

    # another spot higher than  the inner canthus has been found -> on the side of the forehead.
    innercanthus = image1[topleft[1]:bottomleft[1], bottomleft[0]:bottomright[0]]
    bar = image1[topleftbar[1]:bottomleftbar[1], bottomleftbar[0]:bottomrightbar[0]]

    gray_innercanthus = cv2.cvtColor(innercanthus, cv2.COLOR_BGR2GRAY)
    gray_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)

    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal1, maxVal1, minLoc1, maxLoc1) = cv2.minMaxLoc(gray_innercanthus)
    print("InnerCanthus: ", maxVal1)

    cv2.circle(gray_innercanthus, maxLoc1, 5, (255, 0, 0), 2)

    #gray_innercanthus[maxLoc1] = (0,0,255)
    cv2.imshow("Inner Canthus", gray_innercanthus)
    cv2.imshow("Bar", gray_bar)

    xy_coords = np.flip(np.column_stack(np.where(gray_bar == maxVal1)), axis=1)
    cv2.line(gray_bar, (0, xy_coords[0][1]), (gray_bar.shape[1], xy_coords[0][1]), 0, 3)
    cv2.imshow("Bar", gray_bar)

    cv2.imwrite("/Users/amberdebono/PycharmProjects/Prototype/src/GrayBar.png", gray_bar)

    print(xy_coords)

# loop over the frames
while True:
    # reading the frames
    # image = cv2.imread(args["image"])
    frame = cv2.imread(args["image"])
    image1 = cv2.imread(args["image2"])

    # newimage = cv2.resize(image1, (2688, 1520), interpolation=cv2.INTER_AREA)
    # frame = np.hstack((image, newimage))
    frame = perspective_transformation(frame)

    # frame = cv2.resize(frame, (720,1280), interpolation=cv2.INTER_AREA)

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

    # Perspective Transformation is done so that the Optical Image and Thermal Image are aligned

    (locs, preds) = detect_and_predict_mask(frame, net, model)

    for i in range(0, detections.shape[2]):
        # extracting the confidence/probability associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filtering out weak detection by ensuring the 'confidence' is greater than 0.3 in our case
        if confidence < args["confidence"]:
            continue

        # compute the (x,y)-co-ordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")
        faceWidth = endX - startX
        frameWidth = w
        total = (faceWidth / frameWidth) * 100
        rounded = round(total)

        # setting the threshold to 16 for distance
        if rounded < 16:
            cv2.putText(frame, "Too close to camera", (2100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0), 2)
        elif rounded > 16:
            cv2.putText(frame, "Too far from camera", (2100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Ideal Distance reached: " + str(rounded) + "%", (2100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)

        # drawing the bounding face of the face including the probability
        text = "{:.1f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        for rect in rects:

            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                # Light Exposure
                total_pixel = np.size(gray)
                dark_pixel = np.sum(dark_part > 0)
                bright_pixel = np.sum(bright_part > 0)

                if dark_pixel / total_pixel > bright_thres:
                    cv2.putText(frame, "Face is underexposed", (2100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 2)
                elif bright_pixel / total_pixel > dark_thres:
                    cv2.putText(frame, "Face is overexposed", (2100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Good Lighting", (2100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 2)

                LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(frame, shape)
                aligned_face = getaligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

                # Glasses Prediction
                judge = eyeglass(aligned_face)
                if judge:
                    # if glasses are detected then the inner canthus will not be detected and not even the mask.
                    cv2.putText(frame, "Please remove Glasses", (2100, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Glasses detected", (2100, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0), 2)

                    # (x_face + 100, y_face - 10)
                    # (x_face + 100, y_face - 10)

                    # Inner Canthus localisation

                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw them on the image
                    pts = np.array([[shape[21]], [shape[39]], [shape[42]], [shape[22]]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    isClosed = True  # if the polygon is a closed shape
                    print(pts)

                    color = (0, 0, 255)
                    thickness = 2

                    cv2.polylines(frame, [pts], isClosed, color, thickness)
                    cv2.polylines(image1, [pts], isClosed, color, thickness)

                    grayImage = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    cropped = grayImage[rect.top(): rect.bottom(), rect.left(): rect.right()]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cropped)
                    cv2.circle(cropped, maxLoc, 10, 0, 2)
                    print(maxVal)
                    cv2.imshow("Gray Image", cropped)

                    inner_canthus_coords_thermal(image1, shape)

                    # Mask detection

                    # loop over the detected face locations and their corresponding
                    # locations
                    for (box, pred) in zip(locs, preds):
                        # unpack the bounding box and predictions
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        if mask > withoutMask:
                            cv2.putText(frame, "Mask Detected", (2100, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2,
                                        (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "No Mask Detected", (2100, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2,
                                        (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Optical Image", frame)
    cv2.imshow("Thermal Image", image1)
    key = cv2.waitKey(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
