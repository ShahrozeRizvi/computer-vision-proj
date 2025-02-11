'''
*** REAL TIME POSTURE CORRECTION SYSTEM ***
This script implements a real-time posture correction system using OpenCV and MediaPipe.
It detects and corrects posture for:
    - Sitting

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


def process_frame(frame, status):
    """
    Function to process the frame given the status of the processing

    Args:
    frame (np.array): the frame to be processed
    status (str): the status of the processing

    Returns:
    np.array: the processed frame
    """
    # Processing to be made before the frame is passed to the mediapipe pose object
    if status == "pre":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # This is just a optimization for the mediapipe pose object to work faster
        image.flags.writeable = False
    # Processing to be made after the frame is passed to the mediapipe pose object
    if status == "post":
        frame.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return image
def extract_top_landmarks(landmarks, direction, mp_pose):
    """
    Function to extract the (x,y) coordinates of the landmarks of the top part
    of the body given the direction and the mediapipe pose object

    Args:
    landmarks (list): the list of landmarks detected by the mediapipe pose object
    direction (str): the direction of the body part to be extracted
    mp_pose (mediapipe.solutions.pose): the mediapipe pose object

    Returns:
    list: the landmarks of the ear, shoulder and hip
    """
    if direction == "left":
        ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    if direction == "right":
        ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    return ear, shoulder, hip
def compute_time(fps, neck_angle, torso_angle, good_frames, bad_frames):
    """
    Function to compute the time the user has been in a good or bad posture
    given the fps of the camera, the angles of the neck and torso and the
    number of good and bad frames
    The function also compute a guide to correct the posture

    Args:
    fps (int): the fps of the camera
    neck_angle (float): the angle of the neck
    torso_angle (float): the angle of the torso
    good_frames (int): the number of good frames
    bad_frames (int): the number of bad frames

    Returns:
    int: the number of good frames
    int: the number of bad frames
    float: the time the user has been in a good posture
    float: the time the user has been in a bad posture
    str: the guide to correct the posture
    """
    sit_guide = ""
    # Angles values that determine a good sitting posture
    if neck_angle < 70 and torso_angle < 10:
        bad_frames = 0
        good_frames += 1
    else:
        good_frames = 0
        bad_frames += 1
    if neck_angle > 70:
        sit_guide += "Increase the height of the neck\n"
    if torso_angle > 10:
        sit_guide += "Increase the height of the torso\n"
    # The time is computed by multiplying the number of frames by the time of each frame
    good_time = (1/fps) * good_frames
    bad_time = (1/fps) * bad_frames
    return good_frames, bad_frames, good_time, bad_time, sit_guide


# *** POSTURE CORRECTION SYSTEM *** #
def posture_correction(posture, mp_drawing, mp_pose, font, colors):
    """
    Function to correct the posture of the user given the posture to be corrected,
    using the mediapipe pose and drawing objects.

    Args:
    posture (str): the posture to be corrected
    mp_drawing (mediapipe.solutions.drawing_utils): the mediapipe drawing object
    mp_pose (mediapipe.solutions.pose): the mediapipe pose object
    font (int): the font to be used in the frame
    colors (dict): the colors to be used in the frame

    Returns:
    None
    """
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Get the width, height, and fps of the camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    guide = ""
    
    if posture == "sit":
        good_frames = 0
        bad_frames = 0
        good_time = 0
        bad_time = 0
    elif posture not in ["squat", "pushup"]:
        print("Invalid posture")
        return

    # Start the mediapipe pose object
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            _, frame = cap.read()
            frame = process_frame(frame, "pre")
            results = pose.process(frame)
            frame = process_frame(frame, "post")

            try:
                landmarks = results.pose_landmarks.landmark
                
                if posture == "sit":
                    # Extract the required landmarks
                    l_ear, l_shoulder, l_hip = extract_top_landmarks(landmarks, "left", mp_pose)
                    r_ear, r_shoulder, r_hip = extract_top_landmarks(landmarks, "right", mp_pose)

                    # Compute the distance between shoulders
                    offset = find_distance(l_shoulder, r_shoulder, w, h)
                    cv2.rectangle(frame, (w-150,0), (w,40), colors["black"], -1)

                    if offset < 100:
                        cv2.putText(frame, "Aligned", (w-145, 25), font, 0.7, colors["green"], 1, cv2.LINE_AA)

                        # Compute neck and torso angles
                        neck_angle = compute_sit_angles(r_shoulder, r_ear, w, h)
                        torso_angle = compute_sit_angles(r_hip, r_shoulder, w, h)

                        # Display angles on the frame
                        cv2.putText(frame, f"Neck Angle: {round(neck_angle, 2)}",
                                    (int(r_shoulder[0] * w), int(r_shoulder[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Torso Angle: {round(torso_angle, 2)}",
                                    (int(r_hip[0] * w), int(r_hip[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)

                        # Compute posture feedback timing
                        good_frames, bad_frames, good_time, bad_time, guide = compute_time(fps, neck_angle, torso_angle, good_frames, bad_frames)

                        cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)

                        # Display posture time
                        if good_time > 0:
                            cv2.putText(frame, f'Good Posture Time: {round(good_time, 1)}s', (15,12), 
                                        font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, f'Bad Posture Time: {round(bad_time, 1)}s', (15,12), 
                                        font, 0.5, colors["black"], 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Not Aligned", (w-145, 25), font, 0.7, colors["red"], 1, cv2.LINE_AA)
                
                
                cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)

            except:
                pass

            if guide:
                cv2.putText(frame, "GUIDE:", (15,36), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                for i, guide_line in enumerate(guide.split('\n')):
                    cv2.putText(frame, guide_line, (15, 36 + (i+1) * 12), font, 0.5, colors["black"], 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Mediapipe Feed', frame)
            video_writer.write(frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
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


