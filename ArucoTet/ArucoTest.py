
import numpy as np
import pathlib
import os
import cv2
def calibrate_chessboard(dir_path, image_format, square_size, width, height):
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dir_path)
    
    # Iterate through all images
    for fname in images:
        img = cv2.imread(dir_path+"\\"+str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]
def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


# IMAGES_DIR = r'C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\Chessimages'
# IMAGES_FORMAT = '.jpg'
# SQUARE_SIZE = 1.6
# WIDTH = 6
# HEIGHT = 9

# # Calibrate 
# ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(
#     IMAGES_DIR, 
#     IMAGES_FORMAT, 
#     SQUARE_SIZE, 
#     WIDTH, 
#     HEIGHT
# )

# Load coefficients
def scale_image( scale,image):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    return resized_img
mtx, dist = load_coefficients(r'C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\Chessimages\calibration_chessboard.yml')
original = cv2.imread(r'C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\Chessimages\3.jpg')
dst = cv2.undistort(original, mtx, dist, None, None)
cv2.imshow("hey",scale_image(80,dst))
cv2.waitKey(0)
cv2.destroyAllWindows()


