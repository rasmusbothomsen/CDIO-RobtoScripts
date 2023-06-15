import numpy as np
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.best_first import BestFirst
from pathfinding.core import heuristic, diagonal_movement
import math


class NavigationController:
    def __init__(self, image):
        self.image = image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    def rotate_vector(self,vector, angle_deg):
        # Convert angle from degrees to radians
        angle_rad = math.radians(angle_deg)
        
        # Extract the x and y components of the vector
        x, y = vector
        
        # Compute the new x and y components after rotation
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # Return the rotated vector
        return (new_x, new_y)
    def VectorOf2Points(self,v1,v2):
        return ((v2[0]-v1[0]),(v2[1]-v1[1]))

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

    def find_circles(self,image, blue_thresh, red_thresh, green_thresh):
        # Converts image from RGB to grayscale
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.copy()

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

    def create_binary_mesh(self,borderSize):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        image = self.k_means(False)
        self.image = self.expand_red_selection(self.image, borderSize)
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
            if(len(new_dex)<1):
                res = 100
            else:
                res =  sum(tuple(map(lambda i, j: i - j, new_dex[-1], path[idx])))
            if (vector_points[idx] != vector_points[idx + 1]) and res > 10:
                new_dex.append(path[idx])
        new_ar = []
        new_ar.append((start.x,start.y))
        new_ar.extend(new_dex)
        new_ar.append((goal.x,goal.y))
        return new_ar
    def expand_red_selection(self, segmented_image, border_size):
        gray = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(gray, 
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        self.orangCp = orange.copy()
        for contour in contours:
    # Calculate the area of the contour
            area = cv2.contourArea(contour)
            
            # Check if the area is less than 100
            if area < 100:
                # Create a mask for the contour region
                mask = np.zeros_like(gray)
                
                # Set the pixels within the contour region to black
                orange[mask == 255] = [0, 128, 255]  # Set RGB values to black
            if area > 100 and area < 4000:
                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate the coordinates for the square border
                top = max(0, y - border_size)
                bottom = min(segmented_image.shape[0], y + h + border_size)
                left = max(0, x - border_size)
                right = min(segmented_image.shape[1], x + w + border_size)

                # Draw the square border on the image
                cv2.rectangle(orange, (left, top), (right, bottom), (0, 0, 0), -1)
        return orange
    
    def FindVinkel(self,a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return angle

    def FrontAndBack(self,vertices):
        #Finder vinklerne (chatten)
        angles = [self.FindVinkel(vertices[(i + 1) % 3], vertices[i], vertices[(i + 2) % 3]) for i in range(3)]
        #finder den mindste vinkel
        tip_index = np.argmin(angles)
        #Finder koordinat af den
        tip_point = vertices[tip_index]
        #Giver de to sidste vinkel koordinater tilbage, så midten af deres linje kan findes og danne "back" koordinat
        base_points = np.delete(vertices, tip_index, axis=0)

        return tip_point, base_points

    def detectRobot(self,image):
        imagecp = image
        #Skaleret til under 100 giver problemer, fordi billedet er i dårlig kvali, tror gaussian blur driller,
        #Men jeg turde ikke pille for meget ved det
        imagecp = self.image
        imagecp = cv2.cvtColor(imagecp, cv2.COLOR_BGR2RGB)
        image = imagecp.copy()

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangle_contour = []
        for cont in contours:
            perimeter = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.04 * perimeter, True)
            area = cv2.contourArea(cont)
            if len(approx) == 3 and 300 < area < 2000:
                triangle_contour.append(approx)
                break

        #Calculate normalized contour coordinates
        triangle_contour = np.squeeze(triangle_contour)
        triangle_contour = triangle_contour / (image.shape[1], image.shape[0])

        self.triangle_contour = triangle_contour
       
    def getRobotPosition(self):
        triangle_info = {}
         #revert back to pixel coordinates with scaled images
        triangle_contour = self.triangle_contour * (self.image.shape[1], self.image.shape[0])
        triangle_contour = np.expand_dims(triangle_contour, axis=1)
        approx = triangle_contour.astype(np.int32)

        tip_point, base_points = self.FrontAndBack(approx[:, 0])
        mid_base_point = np.mean(base_points, axis=0).astype(int)
        triangle_info['front'] = tuple(tip_point)
        triangle_info['back'] = tuple(mid_base_point)


        return triangle_info
    
    def getRobotAngle(lat1, long1, lat2, long2):
        #lat1 = triangle_info.get("front")[0]
        #long1 = triangle_info.get("front")[1]
        #lat2 = triangle_info.get("back")[0]
        #long2 = triangle_info.get("back")[1]
        # Convert latitude and longitude to radians
        lat1 = math.radians(lat1)
        long1 = math.radians(long1)
        lat2 = math.radians(lat2)
        long2 = math.radians(long2)
  
        # Calculate the bearing
        bearing = math.atan2(
        math.sin(long2 - long1) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1))
  
        # Convert the bearing to degrees
        bearing = math.degrees(bearing)
  
        # Make sure the bearing is positive
        bearing = (bearing + 360) % 360

        radians = bearing * (math.pi/180)
        cos = math.cos(radians)* -1
        sin = math.sin(radians)
        robot_cossin = (cos, sin)        
        return robot_cossin
    
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

        self.image = resized_img

    def find_path(self, start, goal):
       
        
        grid = Grid(matrix=self.binary_image)
        b_first = BestFirst(heuristic=heuristic.euclidean)
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
