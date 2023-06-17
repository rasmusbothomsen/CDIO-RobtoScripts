import cv2
import time

def capture_image(index):
    # Open the camera
    camera = cv2.VideoCapture(0)
    
    # Wait for the camera to warm up
    time.sleep(2)
    
    # Capture an image
    ret, frame = camera.read()
    
    # Save the image with the given index
    filename = f"{index}.jpg"
    cv2.imwrite(filename, frame)
    
    # Release the camera
    camera.release()
    
    print(f"Image {index} captured and saved as {filename}")

# Capture images sequentially with a 2-second delay
for i in range(1, 14):
    capture_image(i)
    time.sleep(2)
