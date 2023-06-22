import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


def detect_green_triangle(image, threshold_low, threshold_high):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper green color thresholds
    lower_green = np.array([threshold_low, 50, 50])
    upper_green = np.array([threshold_high, 255, 255])

    # Create a binary mask of green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours to find the green triangle
    triangle_contour = None
    for cont in contours:
        perimeter = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.03 * perimeter, True)
        area = cv2.contourArea(cont)
        if len(approx) == 3 and 300 < area < 2000:
            triangle_contour = approx
            break

    return triangle_contour


def update_threshold_low(val):
    global threshold_low
    threshold_low = val


def update_threshold_high(val):
    global threshold_high
    threshold_high = val


def on_button_press(event):
    # Check if the left mouse button was clicked
    if event.button == 1:
        # Get the current image coordinates
        x, y = int(event.xdata), int(event.ydata)

        # Check if the click is within the image boundaries
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Display the coordinates
            print("Clicked image coordinates: ({}, {})".format(x, y))


# Load the image
image = cv2.imread(r'C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\Chessimages\4.jpg')

# Create a figure and axes for plotting
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Create sliders for threshold values
ax_threshold_low = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_threshold_low = widgets.Slider(ax_threshold_low, 'Threshold Low', 0, 255, valinit=100)
slider_threshold_low.on_changed(update_threshold_low)

ax_threshold_high = plt.axes([0.25, 0.05, 0.5, 0.03])
slider_threshold_high = widgets.Slider(ax_threshold_high, 'Threshold High', 0, 255, valinit=200)
slider_threshold_high.on_changed(update_threshold_high)

# Detect the green triangle with initial threshold values
threshold_low = slider_threshold_low.val
threshold_high = slider_threshold_high.val
triangle_contour = detect_green_triangle(image, threshold_low, threshold_high)

# Plot the image with detected triangle
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Green Triangle Detection')
plt.axis('off')

# Connect mouse click event handler
cid = fig.canvas.mpl_connect('button_press_event', on_button_press)

# Display the plot
plt.show()

# Disconnect mouse click event handler
fig.canvas.mpl_disconnect(cid)
