import cv2
from skimage import filters, morphology
import numpy as np

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_w, img_h = gray.shape
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # dst = 255 - gray
    dst = gray
    # Smoothing and denoising
    # blur = cv2.medianBlur(gray, 1)
    blur = cv2.medianBlur(dst, 1)
    print('blur = ', blur)

    # Advanced thresholding
    thresh = filters.threshold_otsu(blur)
    print('thresh = ', thresh)
    binary = blur > thresh

    # Morphological operations to enhance structures
    selem = morphology.disk(2)  # Adjust the size of the disk as needed
    cleaned = morphology.opening(binary, selem)

    # Find and draw contours
    contours, _ = cv2.findContours(cleaned.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # result = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    result = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 25000  and w < 1900 and w * h < img_w * img_h: 
            print (w, h)
            # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(result, contour, -1, (255, 0, 0), 2)
            # Find the smallest enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            radius = int(radius)
            print(radius)
            # Draw the circle
            cv2.circle(result, center, radius, (0, 0, 255), 2)
            
            angle = np.deg2rad(45)
            arrow_end_x = int(center[0] + radius * np.cos(angle))
            arrow_end_y = int(center[1] - radius * np.sin(angle))

            # cv2.arrowedLine(result, center, (arrow_end_x, arrow_end_y), (0, 0, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_position_1 = (arrow_end_x + 100 , arrow_end_y - 60)
            text_position_2 = (arrow_end_x + 100 , arrow_end_y - 20)
            font_scale = 1
            font_color = (0, 0, 255)
            line_thickness = 3

            area_true = cv2.contourArea(contour)
            area_circle = np.pi * radius**2
            # 绘制轮廓
            cv2.drawContours(result, contour, -1, (255, 0, 0), 2)

            # 获取轮廓的矩形包围框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算轮廓的中心位置
            cx = x + w // 2
            cy = y + h // 2
            center = (cx, cy)

            # 绘制中心点
            cv2.circle(result, center, 5, (0, 255, 0), -1)  # 绿色小圆点表示中心

            # 绘制面积信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            area_text = "Area   : {:.2f}".format(area_true)
            text_position = (x, y - 10)  # 在轮廓上方显示面积信息
            cv2.putText(result, area_text, text_position, font, 1, (0, 0, 255), 2)

            # 计算并绘制近似半径信息
            radius = int(np.sqrt(area_circle / np.pi))  # 假设轮廓为圆形，计算近似半径
            radius_text = "Approx Radius: {:.2f}".format(radius)
            text_position_2 = (x, y - 30)  # 在面积信息下方显示半径
            cv2.putText(result, radius_text, text_position_2, font, 1, (0, 0, 255), 2)

        # else:
        #     cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Contours", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("result1.jpg", result)

# Replace 'path_to_image.jpg' with your image file path
# process_image('images/2024-01-15/xs2307-8-time-lapse/DSC00025.JPG')
process_image(r"D:\work\3.jpg")