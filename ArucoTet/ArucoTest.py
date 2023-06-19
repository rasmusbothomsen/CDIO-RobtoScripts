import cv2
import numpy as np

image = cv2.imread(r"C:\Users\rasmu\OneDrive\Billeder\Filmrulle\WIN_20230619_09_10_23_Pro.jpg")

detected_markers = np.array([[[302, 74]], [[338, 74]], [[338, 111]], [[302, 111]],
                             [[1463, 87]], [[1498, 87]], [[1498, 121]], [[1463, 121]],
                             [[277, 960]], [[314, 960]], [[314, 977]], [[277, 977]],
                             [[1502, 956]], [[1539, 956]], [[1539, 994]], [[1502, 994]]], dtype=np.float32)

marker_corners = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], 0], [image.shape[1], image.shape[0]]], dtype=np.float32)

# Reshape the marker_corners array
marker_corners = marker_corners.reshape((4, 2))

# Repeat the marker corners for each detected marker
marker_corners = np.repeat(marker_corners[np.newaxis, ...], detected_markers.shape[0], axis=0)

# Calculate the perspective transformation matrix using the marker corners and detected markers
transformation_matrix, _ = cv2.findHomography(detected_markers, marker_corners)

# Apply the perspective transformation to undistort the image
undistorted_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
