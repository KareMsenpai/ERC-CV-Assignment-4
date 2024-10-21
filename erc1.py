import numpy as np
import cv2 as cv 
import mediapipe as mp

# get the mediapipe model for detecting hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

#capture the feed
feed = cv.VideoCapture(0)

while True:
    ret , frame = feed.read() #to read the feed and get the frame
    frame = cv.flip(frame,1)
    rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv.imshow("webcam feed",frame)        

    if cv.waitKey(1) & 0xFF == ord('q'):#gets out of the program when q is pressed
        feed.release()
        cv.destroyAllWindows()
        break


