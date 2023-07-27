import numpy as np
from tqdm import tqdm
import cv2
import os 
import imutils

def crop_image_based_on_extreme_points(input_img):
    # Convert the input image to grayscale and apply Gaussian blur for noise reduction
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    blurred_gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Threshold the blurred image to create a binary image, then perform erosion and dilation to remove noise
    binary_img = cv2.threshold(blurred_gray_img, 45, 255, cv2.THRESH_BINARY)[1]
    binary_img = cv2.erode(binary_img, None, iterations=2)
    binary_img = cv2.dilate(binary_img, None, iterations=2)

    # Find contours in the binary image and obtain the largest contour
    contours = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the extreme points of the largest contour
    leftmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    # Add extra pixels for cropping margin and create a new cropped image
    ADD_PIXELS = 5
    cropped_img = input_img[topmost_point[1]-ADD_PIXELS:bottommost_point[1]+ADD_PIXELS,
                            leftmost_point[0]-ADD_PIXELS:rightmost_point[0]+ADD_PIXELS].copy()

    return cropped_img

def preprocess_and_save_images(input_dir, output_dir):
    # Define the desired image size for resizing
    IMG_SIZE = 256

    # Get a list of directories inside the input directory
    image_directories = os.listdir(input_dir)

    # Loop through each directory in the input directory
    for directory in image_directories:
        # Create the full path to the current input directory
        input_path = os.path.join(input_dir, directory)
        
        # Create the full path to the corresponding output directory
        output_path = os.path.join(output_dir, directory)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Get a list of image files inside the current input directory
        image_files = os.listdir(input_path)

        # Loop through each image file in the current input directory
        for image_file in image_files:
            # Read the image using OpenCV from the current image file
            image = cv2.imread(os.path.join(input_path, image_file))
            
            # Check if the image was read successfully; if not, skip to the next image
            if image is None:
                continue

            # Crop the image using the defined `crop_img()` function
            cropped_image = crop_image_based_on_extreme_points(image)

            # Resize the cropped image to the desired size
            resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))

            # Save the resized image to the output directory with the same file name
            cv2.imwrite(os.path.join(output_path, image_file), resized_image)

if __name__ == "__main__":
    training_input = "brain_tumour/Training"
    testing_input = "brain_tumour/Testing"
    training_output = "brain_tumour/cropped/Training"
    testing_output = "brain_tumour/cropped/Testing"

    preprocess_and_save_images(training_input, training_output)
    preprocess_and_save_images(testing_input, testing_output)
