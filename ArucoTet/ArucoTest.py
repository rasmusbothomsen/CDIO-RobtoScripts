import cv2
import numpy as np


def scale_image(scale,image):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    return resized_img
# Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values

# Load the image
image = cv2.imread(r'ArucoTet\testImage.jpg')
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image, dictionary)

# Assuming that the markers are detected in the order of top-left, top-right, bottom-right, bottom-left
if len(markerCorners) == 4:
    src_pts = np.array([markerCorners[0][0][0], markerCorners[1][0][0], markerCorners[2][0][0], markerCorners[3][0][0]], dtype='float32')

    # Define the destination points. These points are the corners of the image
    dst_pts = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], dtype='float32')

    # Compute Homography
    M, _ = cv2.findHomography(src_pts, dst_pts)

    # Warp source image to destination
    warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    # Show the image
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find 4 markers")