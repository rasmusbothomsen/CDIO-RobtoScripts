import cv2
import numpy as np

def reverse_barrel_distortion(image_path, k):
    # Load the image
    image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the center point of the image
    cx = width / 2
    cy = height / 2

    # Create the undistorted image
    undistorted_image = np.zeros(image.shape, dtype=np.uint8)

    # Reverse the barrel distortion
    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center point
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx ** 2 + dy ** 2)

            # Apply reverse distortion equation
            r_prime = r * (1 + k * r ** 2)

            # Calculate the source coordinates
            src_x = int(cx + dx * (r_prime / r))
            src_y = int(cy + dy * (r_prime / r))

            # Check if the source coordinates are within bounds
            if src_x >= 0 and src_x < width and src_y >= 0 and src_y < height:
                # Copy the pixel from the source to the destination image
                undistorted_image[y, x] = image[src_y, src_x]

    return undistorted_image

# Path to the distorted image
image_path = r"C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\Chessimages\5.jpg"

# Distortion coefficient
k = 0.1

# Reverse the barrel distortion
undistorted_image = reverse_barrel_distortion(image_path, k)

# Display the undistorted image
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
