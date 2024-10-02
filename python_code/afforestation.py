import cv2
import numpy as np


class AfforestationArea:
    def __init__(self, image_path):
        # Load the image
        self.img = cv2.imread(image_path)
        self.hsv = None
        self.masktree = None
        self.maskhouses = None
        self.maskroads = None
        self.maskfeilds = None
        self.res = None

        # If image is successfully loaded, process it
        if self.img is not None:
            self.process_image()

    def apply_filters(self):
        """Applies various filters to the image."""
        # Mean-shift filtering to smooth the image
        shifted = cv2.pyrMeanShiftFiltering(self.img, 7, 30)
        # Convert the filtered image to HSV color space
        self.hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)

    def create_masks(self):
        # a place where we can add HSV color recognition by YOLO

        """Creates masks for different land types based on HSV color ranges."""
        # Define color ranges for different land features in HSV
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

        # Create masks for the different land features
        self.masktree = cv2.inRange(self.hsv, lower_trees, higher_trees)
        self.maskhouses = cv2.inRange(self.hsv, lower_houses, higher_houses)
        self.maskroads = cv2.inRange(self.hsv, lower_roads, higher_roads)
        maskfeilds_houses = cv2.inRange(self.hsv, lower_feilds, higher_feilds)
        blue_limiter = cv2.inRange(self.hsv, lower_feilds_blue, higher_feilds_blue)

        # Combine masks for fields
        self.maskfeilds = maskfeilds_houses

    def process_image(self):
        """Applies the filters and creates masks for the image."""
        self.apply_filters()
        self.create_masks()
        # Apply the field mask to the original image
        self.res = cv2.bitwise_and(self.img, self.img, mask=self.maskfeilds)

    def calculate_area(self):
        """Calculates the area available for afforestation and estimates the number of trees."""
        # Calculate the number of non-zero pixels in the result image
        no_of_non_zero_pixels_rgb = np.count_nonzero(self.res)
        row, col, channels = self.res.shape

        # Calculate the percentage of free land
        percentage_of_land = no_of_non_zero_pixels_rgb / (row * col * channels)

        # Conversion from pixel to cm
        cm_2_pixel = 37.795275591
        row_cm = row / cm_2_pixel
        col_cm = col / cm_2_pixel

        # Calculate total area in cm^2
        tot_pixels = row * col
        tot_area_cm = tot_pixels / (row_cm * col_cm)
        tot_area_cm_land = tot_area_cm * percentage_of_land

        # Conversion from cm^2 to m^2
        tot_area_m_actual_land = tot_area_cm_land * 516.5289256198347

        # Conversion from m^2 to acres
        tot_area_acre_land = tot_area_m_actual_land * 0.000247105

        print(f"Total area in acres: {tot_area_acre_land:.2f}")

        return tot_area_acre_land

    def display_images(self):
        """Displays the processed image and masks."""
        cv2.namedWindow('Filtered Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Field Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)

        cv2.imshow('Filtered Image', self.res)
        cv2.imshow('Field Mask', self.maskfeilds)
        cv2.imshow('Original Image', self.img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
