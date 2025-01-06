import cv2
import dlib
from scipy.spatial import distance as dist
import playsound

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Function to detect mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the three sets of vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])

    # compute the euclidean distance between the horizontal mouth landmarks (x, y)-coordinates
    D = dist.euclidean(mouth[12], mouth[16])

    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)

    # return the mouth aspect ratio
    return mar

# Constants
EYE_AR_THRESH = 0.3  # eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 48  # consecutive frames for which the eye must be below threshold to trigger drowsiness
YAWN_THRESH = 20  # mouth aspect ratio threshold
YAWN_CONSEC_FRAMES = 16  # consecutive frames for which the mouth must be above threshold to trigger drowsiness

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video stream and set the frame dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize counters
eye_counter = 0
yawn_counter = 0

# Loop over frames from the video stream
while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over detected faces
    for face in faces:
        # Determine facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract eye coordinates
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Calculate mouth aspect ratio
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)

        # Average the eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0

        # Check for drowsiness
        if avg_ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play alarm sound
                playsound.playsound("alarm.wav")
        else:
            eye_counter = 0

        if mar > YAWN_THRESH:
            yawn_counter += 1
            if yawn_counter >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play alarm sound
                playsound.playsound("alarm.wav")
        else:
            yawn_counter = 0

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
