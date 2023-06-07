import numpy as np
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.best_first import BestFirst
from pathfinding.core import heuristic, diagonal_movement
from pathfinding.finder import best_first


class NavigationController:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def scale_image(self, scale):
        scale_percent = scale  # percent of original size
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
        resized_img = cv2.resize(resized, new_img_size)
        lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)

        # Split the LAB channels
        l, a, b = cv2.split(lab)

        # Create a CLAHE object and apply it to the L channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge the CLAHE-adjusted L channel with the original A and B channels
        lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert the LAB image back to RGB color space
        self.image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def k_means(self, show_clusters=False):
        np.random.seed(0)
        new_image = self.image
        pixel_values = new_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
        k = 9
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(new_image.shape)

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
            avg_red = (np.sum(centers[x, :red_channel_idx]) - centers[x, red_channel_idx])
            if avg_red > max_mask:
                mask_idx = x
                max_mask = avg_red
            if show_clusters:
                tmpimg = masked_image.copy()
                tmpimg[labels != x] = [0, 0, 0]
                tmpimg = tmpimg.reshape(new_image.shape)
                self.show_image(tmpimg)

        # Set all pixels not belonging to the mask with the most red to black
        masked_image[labels != mask_idx] = [0, 0, 0]

        masked_image = masked_image.reshape(new_image.shape)

        self.image = masked_image

    def find_circles(self, blue_thresh, red_thresh, green_thresh):
        # Converts image from RGB to grayscale
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        image = self.image.copy()

        # Reduces noise by blurring image
        img_blur = cv2.medianBlur(img, 5)

        # HoughCircles is used to find circles in the image.
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            1,
            50,
            param1=50,
            param2=20,
            minRadius=10,
            maxRadius=15,
        )

        circles = np.round(circles[0, :]).astype(np.int32)
        new_circles = list()
        mean_colors = list()

        for idx in range(len(circles)):
            x, y, r = circles[idx]
            roi = image[y - r : y + r, x - r : x + r]
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
            # if all(x > y for x, y in zip(data, [blue_thresh, green_thresh, red_thresh])):
            if np.mean(data) >= red_thresh:
                new_circles.append(circles[idx])
                mean_colors.append([np.mean(data), circles[idx]])
        mean_colors.sort(key=lambda x: x[0])
        orange_ball = mean_colors[0][1]

        if new_circles is not None:
            for (x, y, r) in new_circles:
                cv2.circle(image, (x, y), r, (0, 0, 255), 2)

        return new_circles, image, orange_ball

    def create_binary_mesh(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        image = self.k_means(False)
        self.image = self.expand_red_selection(self.image, 40)
        image_cp = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary_image = np.zeros_like(image_cp)
        binary_image[image_cp != 0] = 1
        self.binary_image = binary_image
        return binary_image

    def find_path_vector_points(self, path, start, goal):
        vector_points = []

        for idx in range(len(path) - 4):
            x = path[idx][0] - path[idx + 4][0]
            y = path[idx][1] - path[idx + 4][1]
            vector_points.append((x, y))

        new_dex = []
        for idx in range(len(vector_points) - 1):
            if vector_points[idx] != vector_points[idx + 1]:
                new_dex.append(path[idx])
        new_ar = []
        new_ar.append((start.x,start.y))
        new_ar.extend(new_dex)
        new_ar.append((goal.x,goal.y))
        return new_ar

    def expand_red_selection(self, segmented_image, border_size):
        # Extract the red channel from the segmented image
        red_channel = segmented_image[:, :, 2]

        # Create a mask based on the red channel
        mask = cv2.threshold(red_channel, 50, 255, cv2.THRESH_BINARY)[1]

        # Expand the mask by the specified number of pixels using dilation
        kernel = np.ones((border_size, border_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Create a new selection based on the expanded mask
        orange = np.zeros_like(segmented_image)
        orange[:, :, 0] = 0
        orange[:, :, 1] = 128
        orange[:, :, 2] = 255
        orange[mask != 0] = segmented_image[mask != 0]

        return orange

    def scale_image(self, scale):
        scale_percent = scale  # percent of original size
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize image
        resized = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
        resized_img = cv2.resize(resized, new_img_size)
        lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)

        # Split the LAB channels
        l, a, b = cv2.split(lab)

        # Create a CLAHE object and apply it to the L channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge the CLAHE-adjusted L channel with the original A and B channels
        lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert the LAB image back to RGB color space
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        self.image = rgb_clahe

    def find_path(self, start, goal):
       
        
        grid = Grid(matrix=self.binary_image)
        b_first = best_first.BestFirst(heuristic=heuristic.euclidean)
        start = grid.node(start[0], start[1])
        end = grid.node(goal[0], goal[1])
        path, runs = b_first.find_path(start, end, grid)
        print(f"length of path {len(path)}")

        new_ar = self.find_path_vector_points(path, start, end)

        return new_ar

    def show_image(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
