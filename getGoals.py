import cv2
from PIL import Image
import numpy as np

Goal = {'big goal': [], 'small goal': []}
img = Image.open("/Users/berfinfloraturan/Downloads/robot/en/RobotBillede3.jpg")  # replace with your image file
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def onclick(event, x, y, flags, param):
    global img, Goal

    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # green circle
        if not Goal['big goal']:
            Goal['big goal'] = [x, y]
            cv2.putText(img, "big goal", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not Goal['small goal']:
            Goal['small goal'] = [x, y]
            cv2.putText(img, "small goal", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("Both big and small goals are already set. Resetting...")
            Goal = {'big goal': [x, y], 'small goal': []}

cv2.namedWindow('image_window')

cv2.setMouseCallback('image_window', onclick)

while True:
    cv2.imshow('image_window', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Print the final coordinates
print(f'Big goal coordinates: {Goal["big goal"]}')
print(f'Small goal coordinates: {Goal["small goal"]}')
