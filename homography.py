import numpy as np
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
import matplotlib.pyplot as plt

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
    max_shift = int(min(h, w)/3)

    # Define source points (image corners)
    corners_old = np.array([
        [0, 0],
        [0, w-1],
        [h-1, w-1],
        [h-1, 0]
    ], dtype=np.float32).reshape(4,2)

    # Perturb the corners to create destination points
    top_left = corners_old[0] + np.random.uniform(10, max_shift, corners_old[0].shape)
    top_right = np.array([corners_old[1,0] + np.random.uniform(10, max_shift), corners_old[1,1] - np.random.uniform(10, max_shift)])
    bottom_right = np.array([corners_old[2,0] - np.random.uniform(10, max_shift), corners_old[2,1] - np.random.uniform(10, max_shift)])
    bottom_left = np.array([corners_old[3,0] - np.random.uniform(10, max_shift), corners_old[3,1] + np.random.uniform(10, max_shift)])

    corners_new = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32).reshape(4,2)

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners_old, corners_new, cv2.RANSAC)
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

    # If eager execution is not enabled, enable it
    tf.config.run_functions_eagerly(True)  # Optional: This ensures eager execution mode is active

    if isinstance(H, tf.Tensor):
        # Ensure eager execution is enabled to evaluate tensors
        H = H[0].numpy() if H is not None else None
    else:
        H = np.array(H, dtype= np.float32)

    h, w = img.shape[:2]
    img = np.array(img).reshape((h, w, 3))
    
    warped_img = cv2.warpPerspective(img, H, (w, h))
    return warped_img


def generate_homography_unwarp(warped_img):
    '''
    Unwarp a warped image
    INPUT: 
        warped_img (np.ndarray): input image

    OUTPUT: 
        H: homography to undo warp
    '''
    img = warped_img.copy()
    mask = (img[:, :, 0] != 0) | (img[:, :, 1] != 0) | (img[:, :, 2] != 0)
    img[mask] = [255, 255, 255]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
    img_gray = cv2.GaussianBlur(img_gray, (11,11), 0) # remove noise for edge detector
    canny = cv2.Canny(img_gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) # dilate to make edges sharper, less points to detect
    contour = np.zeros_like(img)
    c, h = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    region = sorted(c, key = cv2.contourArea, reverse = True)[:5] # best contour is the largest that is still within the bounds
    contour = cv2.drawContours(contour, region, -1, (255,0,0), 3)

    contour = np.zeros_like(img)

    for c in region:
        eps = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, eps, True)
        if len(corners) == 4: # found the correct set of corners
            break

    cv2.drawContours(contour, c, -1, (255, 0, 0), 3)
    cv2.drawContours(contour, corners, -1, (0, 255, 255), 10)

    corners = sorted(np.concatenate(corners).tolist()) # get the corners sorted and in the right shape

    for idx, c in enumerate(corners):
        char = chr(65 + idx)
        cv2.putText(contour, char, tuple(c), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    corners = sort_corners(corners)

    h, w = warped_img.shape[:2]
    dest = np.array([
            [0, 0],
            [0, w-1],
            [h-1, w-1],
            [h-1, 0]
        ], dtype=np.float32).reshape(4,2)
    Hpp = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dest))

    return Hpp

def detect_corners(img):
    '''
    Detect corners in warped image with black space around it from homography using Harris corner detector

    INPUT: 
        img (np.ndarray): Warped input image

    OUTPUT:
        vertices_refined (np.ndarray): the four bounding vertices of image refined by KMeans  
    '''

    copy = img.copy()
    gray = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(np.pad(gray, 20))
    gray[gray > 0] = 255
    dst = cv2.cornerHarris(gray,3,3,0.04)[20:-20, 20:-20]
    
    # Threshold for an optimal value, it may vary depending on the image.
    copy[dst>0.01*dst.max()]=[255,0,0]
    plt.imshow(copy)

    vertices = np.transpose((dst>0.01*dst.max()).nonzero())
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(vertices)
    vertices_refined = np.intp(kmeans.cluster_centers_)
    
    return vertices_refined

def sort_corners(vertices):
    '''
    Sort corners to be compatible with original vertex scheme in homography warpers
    INPUT: 
        vertices (np.ndarray): list of vertices to sort
    
    OUTPUT: 
        sorted_vertices (np.ndarray): list of vertices sorted in clockwise order
    '''
    vertices = np.array(vertices)
    sorted_vertices = np.zeros((4,2), dtype = 'float32')
    s = vertices.sum(axis = 1)
    d = np.diff(vertices, axis = 1)

    sorted_vertices[0] = vertices[np.argmin(s)]
    sorted_vertices[1] = vertices[np.argmin(d)]
    sorted_vertices[2] = vertices[np.argmax(s)]
    sorted_vertices[3] = vertices[np.argmax(d)]

    return sorted_vertices.astype('int').reshape(4,2)



## junk code for corner detection
# # Step 1: Edge detection using Canny
    # image = np.pad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),20)*255
    # image[image > 0] = 255

    # # edges = cv2.Canny(np.uint8(image), 50, 150)

    # # # Step 2: Find contours in the edge-detected image
    # # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # # Step 3: Find the largest contour (by area)
    # # largest_contour = max(contours, key=cv2.contourArea)

    # # # Step 4: Approximate the contour to a quadrilateral
    # # epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # Approximation accuracy
    # # approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # approx = 
    # if len(approx) == 4:  # Ensure it's a quadrilateral
    #     vertices = approx.reshape(4, 2)
    #     return vertices
    # else:
    #     raise ValueError("Unable to find exactly 4 vertices. Adjust parameters.")


## junk code for generate_homography_unwarp
# h, w = img.shape[:2]

# # Define new corners points (fill in black space basically)
# corners_new = np.array([
#     [0, 0],
#     [0, w-1],
#     [h-1, w-1],
#     [h-1, 0]
# ], dtype=np.float32)

# # Find corners in the warped image 
# corners_old = detect_corners(img)
# corners_old = sort_corners(corners_old)

# # Compute the homography matrix
# H, _ = cv2.findHomography(corners_old, corners_new)
# return H