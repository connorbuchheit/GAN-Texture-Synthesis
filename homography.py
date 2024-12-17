import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_homography_warp(img_size):
    """
    Generate a random homography matrix by perturbing image corners.
    
    INPUTS:
        img_size (tuple): Size of the image (height, width).
        max_shift (int): Maximum pixel shift for corner perturbation.
        
    OUTPUT:
        H: A random homography matrix.
    """
    h, w = img_size
    max_shift = min(h, w)/3

    # Define source points (image corners)
    corners_old = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Perturb the corners to create destination points
    corners_new = corners_old + np.random.uniform(0, max_shift, (4, 2))

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners_old, corners_new)
    return H


def apply_homography(img, H):
    """
    Apply a homography transformation to an image, warping it.
    
    INPUTS:
        img (np.ndarray): Input image.
        H (np.ndarray): Homography matrix.
        
    OUTPUT:
        warped_img: Warped image.
    """
    h, w = img.shape[:2]
    warped_img = cv2.warpPerspective(img, H, (w, h))
    return warped_img


def generate_homography_unwarp(img):
    '''
    Unwarp a warped image
    INPUT: 
        img (np.ndarray): input image

    OUTPUT: 
        H: homography to undo warp
    '''
    h, w = img.shape[:2]
    max_shift = min(h, w)/3

    # Define new corners points (fill in black space basically)
    corners_new = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Find corners in the warped image 
    corners_old = detect_corners(img)

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners_old, corners_new)
    return H


# LOOK INTO WRAPPING THE BELOW CODE IN RANSAC TO IMPROVE RESULTS
def detect_corners(image):
    '''
    Detect corners in warped image with black space around it from homography

    INPUT: 
        image (np.ndarray): Warped input image

    OUTPUT:
        vertices (np.ndarray): the four bounding vertices of image  
    '''
    # Step 1: Edge detection using Canny
    image = np.pad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),20)*255
    image[image > 0] = 255

    edges = cv2.Canny(np.uint8(image), 50, 150)

    # Step 2: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Find the largest contour (by area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 4: Approximate the contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # Approximation accuracy
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:  # Ensure it's a quadrilateral
        vertices = approx.reshape(4, 2)
        return vertices
    else:
        raise ValueError("Unable to find exactly 4 vertices. Adjust parameters.")
