import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.best_first import BestFirst
from pathfinding.finder import *
from pathfinding.core import *

# Call the algorithm

def k_means(image,showClusters = False):
    np.random.seed(0)
    newImage = image
    pixel_values = newImage.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    k = 9
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(newImage.shape)


    masked_image = np.copy(segmented_image)
# convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))

    # Get the index of the red channel (assuming RGB color space)
    red_channel_idx = 2

    # color (i.e cluster) to disable
    max_mask = 0.0
    mask_idx = 0

    for x in range(k):
        # Calculate the average red value of the current cluster
        avg_red = (np.sum(centers[x,:red_channel_idx])-centers[x,red_channel_idx])
        if avg_red > max_mask:
            mask_idx = x
            max_mask = avg_red
        if showClusters:
            tmpimg = masked_image.copy()
            tmpimg[labels != x] = [0,0,0]
            tmpimg = tmpimg.reshape(newImage.shape)
            showImage(tmpimg)

    # Set all pixels not belonging to the mask with the most red to black
    masked_image[labels != mask_idx] = [0, 0, 0]

    masked_image = masked_image.reshape(newImage.shape)

    return masked_image

def FindCircles(image,blueTresh, redTresh,greenTresh):

    # Converts image from BGR to grascale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduces noice by blurring image
    img_blur = cv2.medianBlur(img, 5)

    # HoughCircles is used to find circles in the image.
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=20, minRadius=10, maxRadius=15)


    circles = np.round(circles[0, :]).astype(np.int32)
    newCircles = list()
    meanColors = list()

    for indx in range(len(circles)):
        x,y,r = circles[indx]
        roi = image[y - r: y + r, x - r: x + r]
        width, height = roi.shape[:2]
        mask = np.zeros((width, height, 3), roi.dtype)
        cv2.circle(mask, (int(width / 2), int(height / 2)), r, (255, 255, 255), -1)
        dst = cv2.bitwise_and(roi, mask)
        data = []
        for i in range(3):
            channel = dst[:, :, i]
            indices = np.where(channel != 0)[0]
            color = np.mean(channel[indices])
            data.append(int(color))
        # if all(x > y for x, y in zip(data, [blueTresh,greenTresh,redTresh])):
        if np.mean(data)>=redTresh:
            newCircles.append((circles[indx]))
            meanColors.append([np.mean(data),circles[indx]])
    meanColors.sort(key=lambda x: x[0])
    orangeBall = meanColors[0][1]
   
    if newCircles is not None:
        for (x, y, r) in newCircles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
  
    return newCircles, image,orangeBall
def scaleImage(image,scale):
    scale_percent = scale # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)
    lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Create a CLAHE object and apply it to the L channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # Merge the CLAHE-adjusted L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the LAB image back to RGB color space
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return rgb_clahe





def expand_red_selection(segmented_image,bordersize):
    # Extract the red channel from the segmented image
    red_channel = segmented_image[:,:,2]

    # Create a mask based on the red channel
    mask = cv2.threshold(red_channel, 50, 255, cv2.THRESH_BINARY)[1]

    # Expand the mask by 10 pixels using dilation
    kernel = np.ones((bordersize,bordersize), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Create a new selection based on the expanded mask
    orange = np.zeros_like(segmented_image)
    orange[:,:,0] = 0
    orange[:,:,1] = 128
    orange[:,:,2] = 255
    orange[mask != 0] = segmented_image[mask != 0]

    return orange



def showImage(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread(r"NewImages\351286140_643697153896149_4957919446508816481_n.jpg")
image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

image = scaleImage(image,80)
#showImage(image)
circleImage = image.copy()
circles, circleImage,orangeBall = FindCircles(circleImage,130,130,130)
#showImage(circleImage)
cv2.circle(image,(orangeBall[0],orangeBall[1]),orangeBall[2]+3,(255,255,255),-1)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image = k_means(image,False)
image = expand_red_selection(image,40)
imagecp = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
binary_image = np.zeros_like(imagecp)
binary_image[imagecp != 0] = 1


grid = Grid(matrix=binary_image)
bFirst = best_first.BestFirst(heuristic= heuristic.euclidean)

start = grid.node(circles[0][0],circles[0][1])



for indx in range(1,len(circles)):
    grid.cleanup()
    goal = grid.node(circles[indx][0],circles[indx][1])
    path,runs = bFirst.find_path(start,goal,grid)    
    for i in range(len(path) - 1):
        cv2.line(image, path[i], path[i+1], (0, 0, 255), 2)


if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)

cv2.imshow('Path', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
