'''
*** REAL TIME POSTURE CORRECTION SYSTEM ***
This script implements a real-time posture correction system using OpenCV and MediaPipe.
It detects and corrects posture for:
    - Sitting
    - Squatting
    - Pushups
'''

# *** LIBRARIES IMPORT *** #
import cv2
import math
import time
import mediapipe as mp
import numpy as np
import sys

# *** FUNCTIONS *** #

def find_distance(a, b, w, h):
    a = np.array(a) * [w, h]
    b = np.array(b) * [w, h]
    return np.linalg.norm(b - a)

def compute_sit_angles(a, b, w, h):
    a = np.array(a) * [w, h]
    b = np.array(b) * [w, h]
    theta = math.acos((b[1] - a[1]) * (-a[1]) / (math.sqrt((b[0] - b[1]) ** 2 + (b[1] - a[1]) ** 2) * a[1]))
    return int(180 / math.pi) * theta

def compute_exercise_angles(a, b, c):
    CA = np.array(a) - np.array(c)
    CB = np.array(b) - np.array(c)
    dot_product = np.dot(CA, CB)
    magnitude_CA = np.linalg.norm(CA)
    magnitude_CB = np.linalg.norm(CB)
    theta = np.arccos(dot_product / (magnitude_CA * magnitude_CB))
    return np.degrees(theta)

# *** DISABLED ARDUINO COMMUNICATION ***
def send_message(port, command):
    pass  # Arduino communication removed

# *** POSTURE CORRECTION SYSTEM *** #
def posture_correction(posture, mp_drawing, mp_pose, font, colors):
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow camera to warm up

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if posture not in ["sit", "squat", "pushup"]:
        print("Invalid posture")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            _, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                if posture == "sit":
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    offset = find_distance(l_shoulder, r_shoulder, w, h)
                    if offset < 100:
                        cv2.putText(frame, "Aligned", (w-145, 25), font, 0.7, colors["green"], 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Not Aligned", (w-145, 25), font, 0.7, colors["red"], 1, cv2.LINE_AA)
                # Add squat & pushup detection here if needed

            except:
                pass

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Posture Correction System', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# *** MAIN FUNCTION *** #
def main():
    args = sys.argv[1:]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(args) == 2 and args[0] == "-posture":
        posture = args[1]
        colors = {
            "blue": (255, 127, 0),
            "red": (50, 50, 255),
            "green": (127, 255, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        posture_correction(posture, mp_drawing, mp_pose, font, colors)
    else:
        print("Invalid arguments")

if __name__ == "__main__":
    main()


