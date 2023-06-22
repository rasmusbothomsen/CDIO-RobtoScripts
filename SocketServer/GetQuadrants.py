import time
import cv2
import numpy as np

def choose_quadrants():
    Goals = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    result, img = cam.read()
    
    def onclick(event, x, y, flags, param):
        nonlocal Goals, img
        if event == cv2.EVENT_LBUTTONDOWN:
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # green circle
            if not Goals['Q1']:
                Goals['Q1'] = [x, y]
                cv2.putText(img, "Q1", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif not Goals['Q2']:
                Goals['Q2'] = [x, y]
                cv2.putText(img, "Q2", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif not Goals['Q3']:
                Goals['Q3'] = [x, y]
                cv2.putText(img, "Q3", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif not Goals['Q4']:
                Goals['Q4'] = [x, y]
                cv2.putText(img, "Q4", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("All quadrants are already set. Resetting...")
                Goals = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

    cv2.namedWindow('image_window')
    cv2.setMouseCallback('image_window', onclick)

    while True:
        cv2.imshow('image_window', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print(f'Q1: {Goals["Q1"] }')
    print(f'Q2: {Goals["Q2"]}')
    print(f'Q3: {Goals["Q3"] }')
    print(f'Q4: {Goals["Q4"]}')

choose_quadrants()