import cv2
import numpy as np

#Havde kun et billede at teste på, men roterede det, og det virkede stadig
def scaleImage(image, scale):
    scale_percent = scale
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    new_img_size = (resized.shape[1] - (resized.shape[1] % 32), resized.shape[0] - (resized.shape[0] % 32))
    resized_img = cv2.resize(resized, new_img_size)

    return resized_img

#https://itecnote.com/tecnote/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates/
def FindVinkel(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def FrontAndBack(vertices):
    #Finder vinklerne (chatten)
    angles = [FindVinkel(vertices[(i + 1) % 3], vertices[i], vertices[(i + 2) % 3]) for i in range(3)]
    #finder den mindste vinkel
    tip_index = np.argmin(angles)
    #Finder koordinat af den
    tip_point = vertices[tip_index]
    #Giver de to sidste vinkel koordinater tilbage, så midten af deres linje kan findes og danne "back" koordinat
    base_points = np.delete(vertices, tip_index, axis=0)

    return tip_point, base_points

def detectRobot(image_path):
    imagecp = cv2.imread(image_path)
    #Skaleret til under 100 giver problemer, fordi billedet er i dårlig kvali, tror gaussian blur driller,
    #Men jeg turde ikke pille for meget ved det
    image = imagecp.copy()

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_info = {}

    for cont in contours:
        perimeter = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.04 * perimeter, True)
        area = cv2.contourArea(cont)
        if len(approx) == 3 and area>700:
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
            tip_point, base_points = FrontAndBack(approx[:, 0])
            cv2.circle(image, tuple(tip_point), 5, (255, 0, 0), -1)
            cv2.putText(image, 'Front', (tip_point[0], tip_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            #Finder midtpunktet af grundlinjen for trekanten
            mid_base_point = np.mean(base_points, axis=0).astype(int)
            cv2.circle(image, tuple(mid_base_point), 5, (0, 255, 0), -1)
            cv2.putText(image, 'Back', (mid_base_point[0], mid_base_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            triangle_info['front'] = tuple(tip_point)
            triangle_info['back'] = tuple(mid_base_point)
            break

    cv2.imshow("Shapes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return triangle_info

image_path = r"C:\Users\rasmu\OneDrive\Billeder\Filmrulle\WIN_20230614_10_50_09_Pro.jpg"
triangle_info = detectRobot(image_path)
print(triangle_info.get("front"))
print(triangle_info.get("back"))
