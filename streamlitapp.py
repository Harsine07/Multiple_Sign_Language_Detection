import streamlit as st
import cv2
import cv2 as cv
import csv
import mediapipe as mp
import pickle
import copy
import itertools
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from PIL import Image

# --- Dummy Classes for Robustness (in case utility files are missing) ---
class DummyKeyPointClassifier:
    def __call__(self, landmark_list):
        return 0
class CvFpsCalc:
    def __init__(self, buffer_len): pass
    def get(self): return 0.0

# ---------------------------
# GLOBAL SETUP & CACHED RESOURCES
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ISL_alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
BSL_digits_map = {i: str(i) for i in range(10)}
BSL_alphabets_map = {i: ch for i, ch in enumerate(string.ascii_uppercase)}

@st.cache_resource
def load_resources():
    """Loads all heavy models and resources once using Streamlit's cache."""
    resources = {}
    
    # --- ASL Model ---
    try:
        from models.keypoint_classifier.keypoint_classifier import KeyPointClassifier
        ASL_keypoint_classifier = KeyPointClassifier()
        with open("models/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
            ASL_keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        resources['ASL_classifier'] = ASL_keypoint_classifier
        resources['ASL_labels'] = ASL_keypoint_classifier_labels
    except Exception as e:
        st.error(f"ASL Model/Resource Error: {e}")
        resources['ASL_classifier'] = DummyKeyPointClassifier()
        resources['ASL_labels'] = ["A", "B", "C"]

    # --- BSL Models ---
    try:
        with open("./models/british_one_hand_model.pkl", "rb") as f:
            resources['BSL_one'] = pickle.load(f)
        with open("./models/british_two_hand_model.pkl", "rb") as f:
            resources['BSL_two'] = pickle.load(f)
    except Exception as e:
        st.error(f"BSL Model/Resource Error: {e}")
        resources['BSL_one'] = None
        resources['BSL_two'] = None
        
    # --- ISL Model ---
    try:
        resources['ISL_model'] = keras.models.load_model("models/model.h5")
    except Exception as e:
        st.error(f"ISL Model/Resource Error: {e}")
        resources['ISL_model'] = None
        
    return resources

# ---------------------------
# UTILITY FUNCTIONS (Adapted for Streamlit)
# ---------------------------

# The utility functions (draw_landmarks, pre_process_landmark, calc_landmark_list, etc.) 
# from your original code are kept here and reused across models.

def calc_landmark_list(image, landmarks):
    """Calculates pixel coordinates of landmarks."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Normalizes landmarks for classifier input."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    if not temp_landmark_list: return [0.0] * 42 # Handle empty case

    base_x, base_y = temp_landmark_list[0]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
        
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1.0
    max_value = max_value if max_value != 0 else 1.0
    
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def calc_bounding_rect(image, landmarks):
    """Calculates the bounding box coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([[min(int(l.x * image_width), image_width - 1),
                                 min(int(l.y * image_height), image_height - 1)]
                                for l in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_info_text(image, brect, hand_sign_text):
    """Draws the detected text above the bounding box."""
    cv.rectangle(image, (brect[0], brect[1] - 25), (brect[2], brect[1]), (0, 0, 0), -1)
    cv.putText(image, hand_sign_text, (brect[0], brect[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_landmarks(image, landmark_point):
    """Draws hand skeleton and keypoints."""
    if len(landmark_point) == 0: return image
    fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    for finger in fingers:
        for i in range(len(finger) - 1):
            cv.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i + 1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i + 1]]), (255, 255, 255), 2)
    for x, y in landmark_point:
        cv.circle(image, (x, y), 5, (255, 255, 255), -1)
        cv.circle(image, (x, y), 5, (0, 0, 0), 1)
    return image

def draw_fps(image, fps):
    """Draws FPS on the image."""
    cv.putText(image, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

# ---------------------------
# MAIN STREAMLIT APPLICATION
# ---------------------------
def main():
    st.set_page_config(page_title="Sign Language Detector", layout="wide")
    st.title("👋 Real-Time Multi-Model Sign Language Detection")
    st.markdown("Use the **sidebar** to select a model and control the webcam feed.")
    
    models = load_resources()

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    model_options = {
        "1": "American (ASL)", 
        "2": "British (BSL)", 
        "3": "Indian (ISL)"
    }
    
    selected_model_display = st.sidebar.selectbox(
        "Choose Sign Language Model:",
        options=["Select a Model"] + list(model_options.values())
    )
    
    # Map display name back to choice number ('1', '2', or '3')
    choice = next((key for key, value in model_options.items() if value == selected_model_display), None)
    
    st.sidebar.markdown("---")
    start_button = st.sidebar.button("▶️ Start Webcam", type="primary")
    stop_button = st.sidebar.button("🛑 Stop Webcam")
    
    # --- Main Display Area ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Live Feed: {selected_model_display}")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Current Detection")
        status_text = st.empty()
        
    # --- State Management ---
    if 'run' not in st.session_state:
        st.session_state.run = False

    if start_button:
        if not choice or choice == "Select a Model":
            status_text.warning("Please select a model first.")
        else:
            st.session_state.run = True
            
    if stop_button:
        st.session_state.run = False

    # --- Webcam Loop ---
    if st.session_state.run and choice:
        
        # Initialize resources for the loop
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Main detection loop
        while st.session_state.run and cap.isOpened():
            ret, image = cap.read()
            if not ret or image is None or image.size == 0:
                status_text.warning("Failed to read frame from camera. Is it in use by another app?")
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            current_sign_label = "No Hand Detected"
            
            if results.multi_hand_landmarks:
                
                # --- ASL Logic (choice == "1") ---
                if choice == "1":
                    for hand_landmarks in results.multi_hand_landmarks:
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        if models['ASL_classifier']:
                            hand_sign_id = models['ASL_classifier'](pre_processed_landmark_list)
                            current_sign_label = models['ASL_labels'][hand_sign_id]
                            
                            # Drawing specific to ASL
                            debug_image = draw_landmarks(debug_image, landmark_list)
                            debug_image = draw_info_text(debug_image, brect, current_sign_label)

                # --- BSL Logic (choice == "2") ---
                elif choice == "2":
                    landmarks_all = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks_all.append(landmarks)

                    if len(landmarks_all) == 1 and models['BSL_one']:
                        X = np.array(landmarks_all[0]).reshape(1, -1)
                        pred = models['BSL_one'].predict(X)[0]
                        current_sign_label = BSL_digits_map.get(pred, str(pred))
                        cv2.putText(debug_image, f"Digit: {current_sign_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif len(landmarks_all) == 2 and models['BSL_two']:
                        X = np.array(landmarks_all).flatten().reshape(1, -1)
                        pred = models['BSL_two'].predict(X)[0]
                        current_sign_label = BSL_alphabets_map.get(pred, str(pred))
                        cv2.putText(debug_image, f"Letter: {current_sign_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        current_sign_label = f"BSL: Found {len(landmarks_all)} hands. (Needs 1 or 2)"
                         
                # --- ISL Logic (choice == "3") ---
                elif choice == "3":
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed = pre_process_landmark(landmark_list)
                        
                        if models['ISL_model'] and len(pre_processed) == 42:
                            df = pd.DataFrame(pre_processed).transpose()
                            prediction = models['ISL_model'].predict(df, verbose=0)
                            label_index = np.argmax(prediction)
                            current_sign_label = ISL_alphabet[label_index]
                            
                            cv2.putText(debug_image, current_sign_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                            mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        else:
                            current_sign_label = "ISL Model Error / Feature Mismatch"
            

            # Draw FPS and update UI
            fps = cvFpsCalc.get()
            debug_image = draw_fps(debug_image, fps)

            # Convert to RGB for Streamlit and display
            debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(debug_image_rgb, channels="RGB", use_container_width=True)
            status_text.info(f"Detected Sign: **{current_sign_label}**")
        
        # Cleanup when loop breaks
        cap.release()
        hands.close()
        frame_placeholder.empty()
        status_text.info(f"Webcam stopped. Selected Model: **{selected_model_display}**.")
    
    elif not st.session_state.run and choice and choice != "Select a Model":
        status_text.info(f"Model **{selected_model_display}** is loaded. Click 'Start Webcam' to begin.")

if __name__ == "__main__":
    main()