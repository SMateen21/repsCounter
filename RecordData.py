import cv2
import mediapipe as mp
import numpy as np
import CoordinatesToCSV

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

count = 0

cap = cv2.VideoCapture(0)


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Mediapipe Feed', frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.multi_hand_world_landmarks
        except:
            pass

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(100, 30, 80), thickness=2,
                                              circle_radius=4),
                                          mp_drawing.DrawingSpec(
                                              color=(80, 50, 280), thickness=2,
                                              circle_radius=4))

        cv2.imshow('Mediapipe Feed', image)
        k = cv2.waitKey(10)

        if k == ord('q'):
            break
        elif k == -1:
            continue
        elif k == ord('a'):
            # open = 1
            coor_lst = [1]
            for i in range(21):
                landmark_data = [landmarks[0].landmark[i].x,
                                 landmarks[0].landmark[i].y,
                                 landmarks[0].landmark[i].z]
                for j in landmark_data:
                    coor_lst.append(j)
            CoordinatesToCSV.writeLandmarks(coor_lst)
        elif k == ord('s'):
            # close = 0
            coor_lst = [0]
            for i in range(21):
                landmark_data = [landmarks[0].landmark[i].x,
                                 landmarks[0].landmark[i].y,
                                 landmarks[0].landmark[i].z]
                for j in landmark_data:
                    coor_lst.append(j)
            CoordinatesToCSV.writeLandmarks(coor_lst)
        elif k == ord('r'):
            count = 0
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()
