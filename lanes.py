import cv2
import numpy as np
import matplotlib.pyplot as plt

def _make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def _average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = _make_points(image, left_fit_average)
    right_line = _make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def _canny_image(image):

    #  Convert image to grayscale

    lane_image = np.copy(image)  #  We used copy function to copy image. This is done because if we directly assigned image to lane_image then any change in lane_image will also have a change in original image
    gray_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)  #  By default, OpenCV reads images in BGR (Blue, Green, Red) format

    #  Smoothen the Image to reduce noise

    kernel = 5
    blur = cv2.GaussianBlur(gray_image, (kernel,kernel), 0)

    #  Apply Canny Edge Detection

    canny = cv2.Canny(blur, 50, 150)  #  Params: (img, low_thresh, high_thresh)
    return canny

def _display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image

def _region_of_interest(image):

    #  So what we're going to do is take a triangle whose boundaries we defined over here and apply it on the mask, 
    #  such that the area bounded by the polygonal contour will be completely white.
    #  After this we are gonna take bitwise and by which we are gonna left with only the lanes of region of interest.

    height = image.shape[0]
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def main():
    print("What you want to process: ")
    print("1. Image")
    print("2. Video")
    choice = int(input("Enter Your Choice: "))
    if choice == 1:
        image = cv2.imread("test_image.jpg")
        lane_image = np.copy(image)
        canny_image = _canny_image(lane_image)
        cropped_image = _region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap= 5)
        averaged_lines = _average_slope_intercept(lane_image, lines)
        line_image = _display_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        cv2.imshow("Result", combo_image)
        cv2.waitKey(0)
    if choice == 2:
        cap = cv2.VideoCapture("test_video.mp4")
        while(cap.isOpened()):
            _, frame = cap.read()
            canny_image = _canny_image(frame)
            cropped_canny = _region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
            averaged_lines = _average_slope_intercept(frame, lines)
            line_image = _display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
