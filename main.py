import cv2
import numpy as np
import os

def afforestation_area(img_num):

    img = cv2.imread(f'img{img_num}.png')
    shifted = cv2.pyrMeanShiftFiltering(img, 7, 30)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)

    lower_trees = np.array([10, 0, 10])
    higher_trees = np.array([180, 180, 75])

    lower_houses = np.array([90, 10, 100])
    higher_houses = np.array([255, 255, 255])

    lower_roads = np.array([90, 10, 100])
    higher_roads = np.array([100, 100, 100])

    lower_feilds = np.array([0, 20, 100])
    higher_feilds = np.array([50, 255, 255])

    lower_feilds_blue = np.array([0, 80, 100])
    higher_feilds_blue = np.array([255, 250, 255])

    masktree = cv2.inRange(hsv, lower_trees, higher_trees)
    maskhouses = cv2.inRange(hsv, lower_houses, higher_houses)
    maskroads = cv2.inRange(hsv, lower_roads, higher_roads)
    maskfeilds_houses = cv2.inRange(hsv, lower_feilds, higher_feilds)
    blue_limiter = cv2.inRange(hsv, lower_feilds_blue, higher_feilds_blue)
    maskfeilds = maskfeilds_houses
    res = cv2.bitwise_and(img, img, mask=maskfeilds)

    print(res.shape)  # (640, 622, 3)
    print(np.count_nonzero(res))  # 679089

    print("number of pixels", res.size//3)
    tot_pixels = res.size//3
    # print("number of pixels: row x col", res.)

    no_of_non_zero_pixels_rgb = np.count_nonzero(res)
    row, col, channels = res.shape  # 152886
    print("percentage of free land : ", (no_of_non_zero_pixels_rgb /
                                         (row*col*channels)))  # 0.5686369573954984
    percentage_of_land = no_of_non_zero_pixels_rgb/(row*col*channels)

    # https://www.unitconverters.net/typography/centimeter-to-pixel-x.htm
    # says 1 cm = 37.795275591 pixels
    cm_2_pixel = 37.795275591
    print("row in cm ", row/cm_2_pixel)
    print("col in cm ", col/cm_2_pixel)

    row_cm = row/cm_2_pixel
    col_cm = col/cm_2_pixel
    tot_area_cm = tot_pixels/(row_cm*col_cm)
    tot_area_cm_land = tot_area_cm*percentage_of_land

    print("Total area in cm^2 : ", tot_area_cm_land)

    # in google maps 2.2cm = 50m => 1cm = 22.727272727272727m
    # in real life at zoom 18 1cm^2 = (22.727272727272727m)^2
    # = 516.5289256198347 m^2

    # print("Total area in m^2 : ", tot_area_cm_land*(516.5289256198347))
    # tot_area_m_actual_land = tot_area_cm_land*(516.5289256198347)

    # 1 m^2 = 0.000247105 acres :: source Google

    # tot_area_acre_land = tot_area_m_actual_land*0.000247105
    # print("Total area in acres : ", tot_area_acre_land)

    # https://www.treeplantation.com/tree-spacing-calculator.html
    # says if you have 2 ft between rows, and 2ft between
    # trees will can take 10890 trees per acre.

    # number_of_trees = tot_area_acre_land*10890
    # print(f"{round(number_of_trees)} number of trees can be planted in\
    # {tot_area_acre_land} acres.")

    # show the output image

    # res_resized = cv2.resize(res, (640, 480))  # Resize to 640x480, adjust size as needed
    # mask_resized = cv2.resize(maskfeilds, (640, 480))
    # img_resized = cv2.resize(img, (640, 480))
    # hsv_resized = cv2.resize(hsv, (640, 480))

    # Display the resized images
    # cv2.imshow('res', res_resized)
    # cv2.imshow('mask', mask_resized)
    # cv2.imshow('img', img_resized)
    # cv2.imshow('hsv', hsv_resized)

    folder_name = "new_pics"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_path = os.path.join(folder_name, f"hsv_image{img_num}.png")
    cv2.imwrite(file_path, hsv)

    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

    return percentage_of_land



for i in range(1,7):
    print(i)
    afforestation_area(i)
