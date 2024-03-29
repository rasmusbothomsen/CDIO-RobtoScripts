import cv2


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
def undistort(image):
    mtx, dist = load_coefficients(r"C:\Users\rasmu\OneDrive\Skrivebord\CDIO\RobotProject\CDIO-RobtoScripts\ArucoTet\calibration_chessboard.yml")
    original = image
    dst = cv2.undistort(original, mtx, dist, None, None)
    return dst


