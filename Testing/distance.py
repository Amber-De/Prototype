import cv2
import numpy as np
import argparse
import dlib
import logging
from imutils import face_utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# distance from camera to face
known_distance = 30
# width of face
known_width = 14

bright_thres = 0.5
dark_thres = 0.4

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Function to get the focal length - the distance from the camera lens to
# the sensor
def focal_length(measured_distance, real_width, width_in_image):
    focal_Length = (width_in_image * measured_distance) / real_width
    return focal_Length


# distance estimation function
def distance_finder(Focal_Length, real_face_width, face_width_frame):
    distance = (real_face_width * Focal_Length) / face_width_frame
    return distance


def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray_image, 0, 30)
    bright_part = cv2.inRange(gray_image, 220, 255)

    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    print("faces")
    print(faces)
    rects = detector(gray_image, 0)

    total_pixel = np.size(gray_image)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)

    for (x, y, h, w) in faces:
        face_width = w
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if dark_pixel / total_pixel > bright_thres:
            cv2.putText(image, "Face is underexposed", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 1)
        if bright_pixel / total_pixel > dark_thres:
            cv2.putText(image, "Face is overexposed", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 1)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray_image, rect)
            logging.warning(shape)
            logging.warning(rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image

            pts = np.array([[shape[21]], [shape[39]], [shape[42]], [shape[22]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            isClosed = True

            color = (0, 0, 255)
            thickness = 1

            image = cv2.polylines(image, [pts], isClosed, color, thickness)

    return face_width


# getting image from directory
ref_image = cv2.imread("Testing/Ref_image.png")
ref_image_face_width = face_data(ref_image)
focal_length_num = focal_length(known_distance, known_width, ref_image_face_width)

while True:
    # Note: underscore in python can be used as a variable
    _, frame = cap.read()
    face_data_with_frame = face_data(frame)

    # finding the distance - calling the distance function
    if face_data_with_frame != 0:
        dist = distance_finder(focal_length_num, known_width, face_data_with_frame)
        cv2.putText(frame, f"Distance = {dist}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
