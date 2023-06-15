import numpy as np
import cv2

term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_points = np.zeros((6*9,3), np.float32)
obj_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
real_points = []
img_points = []
#HVIS JEG HAVDE 10 BILLEDER STOD DER 1-10 HERINDE, OG JEG NAVNGAV BILLEDERNE 1.JPG, 2.JPG OSV.
chess_images = [1,2,3,4,5,6,7,8,9,10,11]

for name in chess_images:
    chess_img = cv2.imread('/Users/berfinfloraturan/Downloads/Billeder/'+str(name)+'.jpg')
    chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(chess_gray, (9,6), None)
    if ret == True:
        real_points.append(obj_points)
        corners2 = cv2.cornerSubPix(chess_gray,corners, (11,11), (-1,-1), term_criteria)
        img_points.append(corners)

        # draw the corners on the chessboard
        cv2.drawChessboardCorners(chess_img, (9,6), corners2, ret)
        # draw the rectangle around the chessboard
        x1, y1 = int(corners2[0][0][0]), int(corners2[0][0][1])
        x2, y2 = int(corners2[-1][0][0]), int(corners2[-1][0][1])
        cv2.rectangle(chess_img, (x1, y1), (x2, y2), (0, 0, 255), 2)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_points, img_points, chess_gray.shape[::-1], None, None)
#HER INDSÆTTES BILLEDET SOM SKAL UNDISTORTES
img = cv2.imread('/Users/berfinfloraturan/Downloads/Billeder/robot/Robotten.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print("mtx"+str(mtx))
print()
print("dist"+str(dist))
print()
print("newcameramtx"+str(newcameramtx))
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv2.imwrite('undistorted.jpg', dst)
cv2.destroyAllWindows()