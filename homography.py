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
    corners_old, corner_marked_img = detect_shi_tomasi_corners(np.pad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 20))

    plt.imshow(corner_marked_img)

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners_old, corners_new)
    return H

def detect_shi_tomasi_corners(img, max_corners=4, quality_level=0.1, min_distance=None):
    """
    Detect corners in an image using the Shi-Tomasi corner detection method.
    
    INPUTS:
        img (np.ndarray): Input grayscale image.
        max_corners (int): Maximum number of corners to return.
        quality_level (float): Minimum accepted quality of corners (0–1).
        min_distance (int): Minimum distance between detected corners.
        
    OUTPUTS:
        corners: Coordinates of detected corners (x, y).
        corner_marked_img: Image with corners marked.
    """
    # Convert to grayscale if the image is not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if min_distance == None:
        min_distance = min(img.shape[0], img.shape[1])

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    
    # Convert corner points to integer coordinates
    corners = np.int0(corners).reshape(-1, 2)
    
    # Create a copy of the image to mark the corners
    corner_marked_img = img.copy()
    if len(corner_marked_img.shape) == 2:  # Convert grayscale to BGR for visualization
        corner_marked_img = cv2.cvtColor(corner_marked_img, cv2.COLOR_GRAY2BGR)
    
    for x, y in corners:  # Mark the corners in red
        cv2.circle(corner_marked_img, (x, y), 3, (0, 0, 255), -1)
    
    return corners, corner_marked_img