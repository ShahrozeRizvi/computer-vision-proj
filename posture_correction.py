'''
*** REAL TIME POSTURE CORRECTION SYSTEM ***
The following script is the implementation of a posture correction system based on the paper
"Real-Time Workout Posture Correction using OpenCV and MediaPipe" by Yejin Kwon and Dongho Kim 
published on The Journal of Korean Institute of Information Technology in 2022

The system can be used to detect and correct the posture of the user either while performing squat
or push-up exercises or while sitting. The system uses the MediaPipe library to detect the user's
key points and then computes the angles and distances between them to determine the correctness of
the posture. In the case of the sitting posture the system also uses a serial connection to an 
Arduino to send feedback to the user in the form of a green light if the posture is correct, a red
light if the posture is incorrect and a yellow light if the user is not aligned with the camera.

The system can be run from the command line by passing the argument "-posture" followed by the
posture to be corrected. The system can correct the following postures:
    - sit
    - squat
    - pushup
'''


# *** LIBRARIES IMPORT *** #
import cv2
import math
import serial
import serial.tools.list_ports as port_list
import time
import mediapipe as mp
import numpy as np
import sys

# *** FUNCTIONS *** #

def find_distance(a, b, w, h):
    """
    Function to compute the distance between two points a and b
    given the width and height of the frame in order to scale the
    points to the frame's dimensions

    Args:
    a (list): the first point
    b (list): the second point
    w (int): the width of the frame
    h (int): the height of the frame

    Returns:
    float: the distance between the two points
    """
    a = np.array(a)
    b = np.array(b)
    a[0] = a[0] * w
    a[1] = a[1] * h
    b[0] = b[0] * w
    b[1] = b[1] * h
    dist = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    return dist

def compute_sit_angles(a, b, w, h):
    """
    Function to compute the angle between two points a and b
    given the width and height of the frame in order to scale the
    points to the frame's dimensions

    Args:
    a (list): the first point
    b (list): the second point
    w (int): the width of the frame
    h (int): the height of the frame

    Returns:
    float: the angle between the two points
    """
    a = np.array(a)
    b = np.array(b)
    # scale the points to the frame's dimensions
    a[0] = a[0] * w
    a[1] = a[1] * h
    b[0] = b[0] * w
    b[1] = b[1] * h
    # calculate the angle between the two points
    theta = math.acos((b[1] - a[1]) * (-a[1]) / (math.sqrt((b[0] - b[1]) ** 2 + (b[1] - a[1]) ** 2) * a[1]))
    # convert the angle from radians to degrees
    degree = int(180/math.pi) * theta
    return degree

def compute_exercise_angles(a, b, c):
    """
    Function to compute the angle between three points a, b and c
    By using the dot product of the vectors CA and CB and the magnitudes
    of the vectors CA and CB the scaling is not necessary

    Args:
    a (list): the first point
    b (list): the second point
    c (list): the third point

    Returns:
    float: the angle between the three points
    """
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    # Calculate vectors CA and CB
    CA = np.array([x1-x3, y1-y3, z1-z3])
    CB = np.array([x2-x3, y2-y3, z2-z3])
    # Calculate dot product of CA and CB
    dot_product = np.dot(CA, CB)
    # Calculate magnitudes of vectors CA and CB
    magnitude_CA = np.linalg.norm(CA)
    magnitude_CB = np.linalg.norm(CB)
    # Calculate angle theta using the dot product
    theta = np.arccos(dot_product / (magnitude_CA * magnitude_CB))
    # Convert angle from radians to degrees
    theta_degrees = np.degrees(theta)
    return theta_degrees

def send_message(port, command):
    """
    Function to send a message to the serial port

    Args:
    port (serial.Serial): the serial port
    command (str): the message to be sent

    Returns:
    None
    """
    # If the command is "G" the message sent is the one needed to communicate a good posture
    if command =="G":
        port.write(b'G') 
    # If the command is "R" the message sent is the one needed to communicate a bad posture 
    elif command =="R":
        port.write(b'R')
    # If the command is "Y" the message sent is the one needed to communicate that the user
    # is not aligned with the camera
    elif command =="Y":
        port.write(b'Y')
    # If the command is "V" the message sent is the one needed to communicate that the system
    # is still computing the posture
    elif command =="V":
        port.write(b'V')
    # If the command is "O" the message sent is the one needed to communicate that the system
    # has been stopped
    elif command =="O":
        port.write(b'O') 
    return

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

def extract_squat_landmarks(landmarks, direction, mp_pose):
    """
    Function to extract the (x,y,z) coordinates of the landmarks of the squat exercise
    given the direction and the mediapipe pose object

    Args:
    landmarks (list): the list of landmarks detected by the mediapipe pose object
    direction (str): the direction of the body part to be extracted
    mp_pose (mediapipe.solutions.pose): the mediapipe pose object

    Returns:
    list: the landmarks of the knee, hip, shoulder and foot
    """
    if direction == "left":
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    if direction == "right":
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    return knee, hip, shoulder, foot

def extract_push_up_landmarks(landmarks, direction, mp_pose):
    """
    Function to extract the (x,y,z) coordinates of the landmarks of the push-up exercise
    given the direction and the mediapipe pose object

    Args:
    landmarks (list): the list of landmarks detected by the mediapipe pose object
    direction (str): the direction of the body part to be extracted
    mp_pose (mediapipe.solutions.pose): the mediapipe pose object

    Returns:
    list: the landmarks of the wrist, elbow, shoulder, ankle and hip
    """
    if direction == "left":
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    if direction == "right":
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    return wrist, elbow, shoulder, ankle, hip

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

def compute_squat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_foot, r_foot):
    """
    Function to compute the angles and distances needed to determine
    the correctness of the squat posture

    Args:
    l_shoulder (list): the left shoulder landmark
    r_shoulder (list): the right shoulder landmark
    l_hip (list): the left hip landmark
    r_hip (list): the right hip landmark
    l_knee (list): the left knee landmark
    r_knee (list): the right knee landmark
    l_foot (list): the left foot landmark
    r_foot (list): the right foot landmark

    Returns:
    float: the angle of the hip
    float: the height between the left knee and hip
    float: the height between the right knee and hip
    float: the width between the left foot and knee
    float: the width between the right foot and knee
    """
    # condition 1 - The Hip Angle -
    l_hip_angle = compute_exercise_angles(l_knee, l_hip, l_shoulder)
    r_hip_angle = compute_exercise_angles(r_knee, r_hip, r_shoulder)
    hip_angle = (l_hip_angle + r_hip_angle) / 2

    # condition 2 - The Knee-Hip Height -
    def compute_knee_hip_height(knee, hip):
        """
        Function to compute the height between the knee and hip

        Args:
        knee (list): the knee landmark
        hip (list): the hip landmark

        Returns:
        float: the height between the knee and hip
        """
        _, knee_y, _ = knee
        _, hip_y, _ = hip
        return round(abs(knee_y - hip_y), 3)
    l_knee_hip_height = compute_knee_hip_height(l_knee, l_hip)
    r_knee_hip_height = compute_knee_hip_height(r_knee, r_hip)

    # condition 3 - The Foot-Knee Width -
    def compute_foot_knee_width(foot, knee):
        """
        Function to compute the width between the foot and knee

        Args:
        foot (list): the foot landmark
        knee (list): the knee landmark

        Returns:
        float: the width between the foot and knee
        """
        foot_x, _, _ = foot
        knee_x, _, _ = knee
        return round(abs(foot_x - knee_x), 3)
    l_foot_knee_width = compute_foot_knee_width(l_foot, l_knee)
    r_foot_knee_width = compute_foot_knee_width(r_foot, r_knee)

    return hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width

def isCorrectSquat(hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width):
    """
    Function to determine if the squat posture is correct given the angles and distances
    computed in the previous function

    Args:
    hip_angle (float): the angle of the hip
    l_knee_hip_height (float): the height between the left knee and hip
    r_knee_hip_height (float): the height between the right knee and hip
    l_foot_knee_width (float): the width between the left foot and knee
    r_foot_knee_width (float): the width between the right foot and knee

    Returns:
    bool: the correctness of the squat posture
    str: the guide to correct the posture
    """
    squat_guide = ""
    is_squat_correct = False
    # The logic and of conditions 1, 2 and 3
    are_conditions_met = (60 <= hip_angle <= 120 and l_knee_hip_height <= 0.2 and r_knee_hip_height <= 0.2 and l_foot_knee_width <= 0.1 and r_foot_knee_width <= 0.1)

    if are_conditions_met:
        is_squat_correct = True
    elif hip_angle < 40:
        # In this case the user is in the stand position reached before or after the squat
        is_squat_correct = None
    else:
        # build guide given the wrong conditions
        if 40 <= hip_angle < 60:
            squat_guide += "Increase the height of the hip\n"
        if hip_angle > 120:
            squat_guide += "Decrease the height of the hip\n"
        if l_knee_hip_height > 0.2 or r_knee_hip_height > 0.2:
            squat_guide += "Keep the thigh horizontal to the floor\n"
        if l_foot_knee_width > 0.1 or r_foot_knee_width > 0.1:
            squat_guide += "Do not exceed the tip of the toe with your knee\n"
    
    return is_squat_correct, squat_guide

def compute_push_up(l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle):
    """
    Function to compute the angles and distances needed to determine
    the correctness of the push-up posture

    Args:
    l_shoulder (list): the left shoulder landmark
    r_shoulder (list): the right shoulder landmark
    l_elbow (list): the left elbow landmark
    r_elbow (list): the right elbow landmark
    l_wrist (list): the left wrist landmark
    r_wrist (list): the right wrist landmark
    l_hip (list): the left hip landmark
    r_hip (list): the right hip landmark
    l_ankle (list): the left ankle landmark
    r_ankle (list): the right ankle landmark

    Returns:
    float: the angle of the elbow
    float: the angle of the body
    """
    # Condition 1 - The Elbow Angle -
    l_elbow_angle = compute_exercise_angles(l_wrist, l_elbow, l_shoulder)
    r_elbow_angle = compute_exercise_angles(r_wrist, r_elbow, r_shoulder)
    elbow_angle = (l_elbow_angle + r_elbow_angle) / 2

    # Condition 2 - The Body Angle -
    l_body_angle = compute_exercise_angles(l_ankle, l_hip, l_shoulder)
    r_body_angle = compute_exercise_angles(r_ankle, r_hip, r_shoulder)
    body_angle = (l_body_angle + r_body_angle) / 2
    return elbow_angle, body_angle

def isCorrectPushUp(elbow_angle, body_angle):
    """
    Function to determine if the push-up posture is correct given the angles computed
    in the previous function

    Args:
    elbow_angle (float): the angle of the elbow
    body_angle (float): the angle of the body

    Returns:
    bool: the correctness of the push-up posture
    str: the guide to correct the posture
    """
    push_up_guide = ""
    is_push_up_correct = False

    if 70 <= elbow_angle <= 100 and 160 <= body_angle <= 200:
        is_push_up_correct = True
    elif body_angle < 130:
        # In this case the user either still have to start the push-up or has already finished it
        is_push_up_correct = None
    else:
        if elbow_angle < 70:
            push_up_guide += "Increase the height of the elbow\n"
        if elbow_angle > 100:
            push_up_guide += "Decrease the height of the elbow\n"
        if 130 <= body_angle < 160:
            push_up_guide += "Increase the height of the body\n"
        if body_angle > 200:
            push_up_guide += "Decrease the height of the body\n"
    return is_push_up_correct, push_up_guide

def posture_correction(posture, mp_drawing, mp_pose, font, colors, port):
    """
    Function to correct the posture of the user given the posture to be corrected,
    the mediapipe pose and drawing objects and the serial port to communicate with

    Args:
    posture (str): the posture to be corrected
    mp_drawing (mediapipe.solutions.drawing_utils): the mediapipe drawing object
    mp_pose (mediapipe.solutions.pose): the mediapipe pose object
    font (int): the font to be used in the frame
    colors (dict): the colors to be used in the frame
    port (str): the serial port to communicate with

    Returns:
    None
    """
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Open the serial port
    arduino = serial.Serial(port, 9600)
    # Wait for the serial port to be ready
    time.sleep(2)
    
    # Get the width, height and fps of the camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer to save the output
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    # Initialize the variables to compute the feedback to the user
    guide = ""
    if posture == "sit":
        good_frames = 0
        bad_frames = 0
        good_time = 0
        bad_time = 0
    elif posture == "squat" or posture == "pushup":
        pass
    # If the posture chosen by terminal by the user is not valid the system stops
    else:
        print("Invalid posture")
        return
    # Start the mediapipe pose object
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Start the loop to process the frames
        while cap.isOpened():
            _, frame = cap.read()
            frame = process_frame(frame, "pre")
            results = pose.process(frame)
            frame = process_frame(frame, "post")
            try:
                # Extract the landmarks of the user
                landmarks = results.pose_landmarks.landmark
                if posture == "sit":
                    # Extract the needed landmarks of the user for the sitting posture
                    l_ear, l_shoulder, l_hip = extract_top_landmarks(landmarks, "left", mp_pose)
                    r_ear, r_shoulder, r_hip = extract_top_landmarks(landmarks, "right", mp_pose)
                    # Compute the distance between the shoulders
                    offset = find_distance(l_shoulder, r_shoulder, w, h)
                    cv2.rectangle(frame, (w-150,0), (w,40), colors["black"], -1)
                    # If the user is correctly aligned with the camera, meaning that is by the side
                    if offset < 100:
                        cv2.putText(frame, "Aligned", (w-145, 25), font, 0.7, colors["green"], 1, cv2.LINE_AA)
                        # Compute the angles of the neck and torso
                        neck_angle = compute_sit_angles(r_shoulder, r_ear, w, h)
                        torso_angle = compute_sit_angles(r_hip, r_shoulder, w, h)
                        # put on the frame the neck angle at the height of the right shoulder
                        cv2.putText(frame, "Neck Angle: " + str(round(neck_angle, 2)),
                                (int(r_shoulder[0] * w), int(r_shoulder[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the torso angle at the height of the right hip
                        cv2.putText(frame, "Torso Angle: " + str(round(torso_angle, 2)),
                                (int(r_hip[0] * w), int(r_hip[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        
                        # Compute the time the user has been in a good or bad posture
                        good_frames, bad_frames, good_time, bad_time, guide = compute_time(fps, neck_angle, torso_angle, good_frames, bad_frames)                

                        cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)
                        # Display the time the user has been in a good or bad posture
                        if good_time > 0:
                            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                            cv2.putText(frame, time_string_good, (15,12), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        else:
                            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                            cv2.putText(frame, time_string_bad, (15,12), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)

                        # Send the feedback to the user through the serial port
                        if good_time > 10:
                            send_message(arduino, "G")
                        elif bad_time > 10:
                            send_message(arduino, "R")
                        else:
                            send_message(arduino, "V")
                    # If the user is not aligned with the camera   
                    else:
                        cv2.putText(frame, "Not Aligned", (w-145, 25), font, 0.7, colors["red"], 1, cv2.LINE_AA)
                        send_message(arduino, "Y")
                else:
                    if posture == "squat":
                        # Extract the needed landmarks of the user for the squat posture
                        l_knee, l_hip, l_shoulder, l_foot = extract_squat_landmarks(landmarks, "left", mp_pose)
                        r_knee, r_hip, r_shoulder, r_foot = extract_squat_landmarks(landmarks, "right", mp_pose)
                        # Compute the angles and distances needed to determine the correctness of the squat posture
                        hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width = compute_squat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_foot, r_foot)
                        # put on the frame the hip angle at the height of the left hip
                        cv2.putText(frame, "Hip Angle: " + str(round(hip_angle, 2)),
                                (int(l_hip[0] * w), int(l_hip[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the knee-hip height at the height of the left knee
                        cv2.putText(frame, "Knee-Hip Height: " + str(round(l_knee_hip_height, 2)),
                                (int(l_knee[0] * w), int(l_knee[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the foot-knee width at the height of the left foot
                        cv2.putText(frame, "Foot-Knee Width: " + str(round(l_foot_knee_width, 2)),
                                (int(l_foot[0] * w), int(l_foot[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        is_exercise_performed, guide = isCorrectSquat(hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width)
                    if posture == "pushup":
                        # Extract the needed landmarks of the user for the push-up posture
                        l_wrist, l_elbow, l_shoulder, l_ankle, l_hip = extract_push_up_landmarks(landmarks, "left", mp_pose)
                        r_wrist, r_elbow, r_shoulder, r_ankle, r_hip = extract_push_up_landmarks(landmarks, "right", mp_pose)
                        # Compute the angles needed to determine the correctness of the push-up posture
                        elbow_angle, body_angle = compute_push_up(l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle)
                        # put on the frame the elbow angle at the height of the left elbow
                        cv2.putText(frame, "Elbow Angle: " + str(round(elbow_angle, 2)),
                                (int(l_elbow[0] * w), int(l_elbow[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the body angle at the height of the left ankle
                        cv2.putText(frame, "Body Angle: " + str(round(body_angle, 2)),
                                (int(l_ankle[0] * w), int(l_ankle[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        is_exercise_performed, guide = isCorrectPushUp(elbow_angle, body_angle)

                    cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)
            except:
                pass

            # If a guide containing the corrections to be made is available, display it on the frame
            if guide != "":
                cv2.putText(frame, "GUIDE: ", 
                                (15,36), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)
                for i, guide_line in enumerate(guide.split('\n')):
                    t = i+1
                    cv2.putText(frame, guide_line, 
                        (15,36 + t*12), 
                        font, 0.5, colors["black"], 1, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=colors["pink"], thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=colors["blue"], thickness=2, circle_radius=2) 
                                    )               
            cv2.imshow('Mediapipe Feed', frame)
            # Save the frame to the output video
            video_writer.write(frame)
            # If the user presses 'q' the system stops
            if cv2.waitKey(10) & 0xFF == ord('q'):
                # Send the message to the serial port to stop the system
                send_message(arduino, "O")
                break
        # Close the camera and the video writer
        cap.release()
        video_writer.release()
        # Close the window
        cv2.destroyAllWindows()
        # Close the serial port
        arduino.close()
    return


def main():
    args = sys.argv[1:]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(args) == 2 and args[0] == "-posture":
        posture = args[1]

        colors = {
            "blue": (255, 127, 0),
            "red": (50, 50, 255),
            "green": (127, 255, 0),
            "dark_blue": (127, 20, 0),
            "light_green": (127, 233, 100),
            "yellow": (0, 255, 255),
            "pink": (255, 0, 255),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "light_cyan" : (245,117,16)
        }

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        port = list(port_list.comports())
        port = str(port[0].device)

        posture_correction(posture, mp_drawing, mp_pose, font, colors, port)
    else:
        print("Invalid arguments")
    return

if __name__ == "__main__":
    main()