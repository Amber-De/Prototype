import dlib
import cv2
import numpy as np
import argparse
from imutils import face_utils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initializing dlib's face detector (HOG-based) and then creating
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Declaring the least bright/dark it can be
bright_thres = 0.5
dark_thres = 0.4


# coordinates in a np array function
def landmarks_to_np(landmarks, dtype="int"):
    num =  landmarks.num_parts

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
    print(EYE_LEFT_OUTER)
    EYE_LEFT_INNER = landmarks[39]
    print(EYE_LEFT_INNER)
    EYE_RIGHT_OUTER = landmarks[42]
    print(EYE_RIGHT_OUTER)
    EYE_RIGHT_INNER = landmarks[45]
    print(EYE_RIGHT_INNER)

    x = ((landmarks[36:40]).T)[0]
    print(x)
    y = ((landmarks[42:46]).T)[1]
    print(y)
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

    print(measure)

    if measure > 0.15:
        judge = True
    else:
        judge = False
    print(judge)
    return judge


cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    rects = detector(gray, 1)

    for (x, y, h, w) in faces:
        # Light Exposure
        total_pixel = np.size(gray)
        dark_pixel = np.sum(dark_part > 0)
        bright_pixel = np.sum(bright_part > 0)

        for rect in rects:
            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face

            cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

            if dark_pixel / total_pixel > bright_thres:
                cv2.putText(img, "Face is underexposed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 1)
            if bright_pixel / total_pixel > dark_thres:
                cv2.putText(img, "Face is overexposed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 1)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, shape)
        aligned_face = getaligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

        # Glasses Prediction
        judge = eyeglass(aligned_face)
        if judge:
            cv2.putText(img, "Please remove Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Glasses detected", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            # Inner Canthus localisation
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            pts = np.array([[shape[21]], [shape[39]], [shape[42]], [shape[22]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            isClosed = True

            color = (0, 0, 255)
            thickness = 1

            image = cv2.polylines(img, [pts], isClosed, color, thickness)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
