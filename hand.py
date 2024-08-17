import mediapipe as mp
import cv2
import uuid
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

## camera setup
cap = cv2.VideoCapture(0)   ##0 depends on different applications
with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret , frame = cap.read()

        # bgr into rgb
        image = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)

        # DISABL THE COPYUING OF IMAGE
        image.flags.writeable = False

        # DETECTING THE IMAGE
        results = hands.process(image)

        # ENABLING THE COPYING OF IMAGE
        image.flags.writeable = True

        # RGB INTO BGR FOR THE OUTPUT
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results)

        if results.multi_hand_landmarks :
            for num , hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image , hand , mp_hands.HAND_CONNECTIONS)

        # Images storing
        cv2.imwrite(os.path.join('Output images', '{}.jpg'.format(uuid.uuid1())),image)

        
        cv2.imshow('hand tracking',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
