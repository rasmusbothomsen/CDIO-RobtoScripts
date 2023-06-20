import time
import cv2
import numpy as np

def choose_goals():
    Goals = {'big_goal': [], 'small_goal': []}
    cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(2)
    result, img = cam.read()
    
    def onclick(event, x, y, flags, param):
        nonlocal Goals, img
        if event == cv2.EVENT_LBUTTONDOWN:
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # green circle
            if not Goals['big_goal']:
                Goals['big_goal'] = [x, y]
                cv2.putText(img, "big_goal", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif not Goals['small_goal']:
                Goals['small_goal'] = [x, y]
                cv2.putText(img, "small_goal", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("Both big and small goals are already set. Resetting...")
                Goals = {'big_goal': [x, y], 'small_goal': []}

    cv2.namedWindow('image_window')
    cv2.setMouseCallback('image_window', onclick)

    while True:
        cv2.imshow('image_window', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print(f'Big goal coordinates: {Goals["big_goal"] }')
    print(f'Small goal coordinates: {Goals["small_goal"]}')

choose_goals()