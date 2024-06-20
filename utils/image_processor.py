import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import cmp_to_key
import matplotlib.pyplot as plt
from scipy.spatial import distance





def import_court(court_loc = 'court_reference.png',gray = None):
    """
    import the court reference image
    """
    return cv2.imread(court_loc) if gray is None else cv2.imread(court_loc, 0) 
    
def court_borders(court_reference,court_factor = 1):
    """
    calculates the location of the border points based on image size
    """
    width = court_reference.shape[1]
    height = court_reference.shape[0]
    court_borders = np.asarray([[0, 0], [width/court_factor, 0], [width/court_factor, height*court_factor], [0, height*court_factor]])
    return  np.float32(court_borders).reshape(-1,1,2)

def gray_scale(image,dialation=3,kernel_size=3):
    """
    creates a threshold image of the original for homography matrix scoring
    """
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.dilate(gray,kernel,iterations=dialation)
    gray[gray > 0] = 1 
    return gray

def get_intersection_1(a1, a2, b1, b2):
    """ 
    returns intersection point between 2 lines
    a1,a2 points of first line
    a3,a4 points of second line
    """
    v_stacked = np.vstack([a1,a2,b1,b2])     
    h_stacked = np.hstack((v_stacked, np.ones((4, 1))))
    
    line_1 = np.cross(h_stacked[0], h_stacked[1])         
    line_2 = np.cross(h_stacked[2], h_stacked[3])           
    x, y, z = np.cross(line_1, line_2)
    if z == 0:  # no intersection
        return (float('inf'), float('inf'))
    x = x/z
    y = y/z
    return (x, y)


def midpoint(line):
    """ 
    finds midpoint of a line
    """
    return ((line[0][0]+line[1][0])/2, (line[0][1]+line[1][1])/2)

def find_line_eq (line,im):
    """ 
    given a line, the function can find the equation
    and extends that line to the borders of the image 
    """
    x_f = 0
    x_e = im.shape[1]
    y_f = 0
    y_e = im.shape[0]
    y_coords, x_coords = zip(*line)
   
   
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords,rcond=None)[0]
   
    theta = math.degrees(math.atan(m))
    
    x0 = (y_f-c)/m
    x1 = (y_e-c)/m

    if x0 > x_e or x0< x_f:
        x0 = np.clip(x0,0,x_e)
        y0 = (m * x0) + c
    else:
        y0 = y_f

    if x1 > x_e or x1< x_f:
        x1 = np.clip(x1,0,x_e)
        y1 = (m * x1) + c
    else:
        y1 = y_e

    return [[y0,x0],[y1,x1]] , theta



def order_points(pts):
    """ 
    To order the 4 border points in a clock-wise fasion
    starting from top left point
    """
    x_sort = pts[np.argsort(pts[:, 0]), :]
    left_x = x_sort[:2, :]
    right_x = x_sort[2:, :]
    left_x = left_x[np.argsort(left_x[:, 1]), :]
    (tl, bl) = left_x
    dist = distance.cdist(tl[np.newaxis], right_x, "euclidean")[0]
    (br, tr) = right_x[np.argsort(dist)[::-1], :]
    return np.array([tl,bl,br,tr], dtype="float32")


def homography_scorer(matrix,court_reference,im,gray):
    """
    Calculate score for homography matrix by comparing the
    result of homography warp to the thresholded original image
    """
    court = cv2.warpPerspective(court_reference, matrix, (im.shape[1],im.shape[0]))
    court[court > 0] = 1    

    return court



def get_corners(H, image_size):
    """
    get the location of new corners of the image 
    after applying the homography matrix
    """    
    limit_y = float(image_size[0])
    limit_x = float(image_size[1])
    # Apply H to to find new image bounds
    tr  = np.dot(H, np.array([0.0,      limit_y, 1.0]).flatten()) # new top left
    br  = np.dot(H, np.array([limit_x,  limit_y, 1.0]).flatten()) # new bottom right
    bl  = np.dot(H, np.array([limit_x,      0.0, 1.0]).flatten()) # new bottom left
    matrix_corners = [tr,br,bl]
    
    for pt in matrix_corners:
        if pt[2] == 0:
            return None

    return matrix_corners

def scale_matrix(homography_matrix, image_size): 
    """
    adjusts the scaling of the matrix to make sure all
    corners of the new warped image is within frame
    """    
    # Get new image corners
    matrix_corners = get_corners(homography_matrix, image_size)

    # don't scale if a point is at infinity
    if matrix_corners is None:
        print("scaling cannot be applied")
        return homography_matrix
        
    scale = [max([matrix_corners[j][i] / matrix_corners[j][2] for j in range(len(matrix_corners))])/float(image_size[i]) for i in range(2)]
    scale = max(scale)
    # apply a maximum scale of 5
    scale = min(scale,5)
    print("Scaling factor:{}".format(scale))
    scaling_matrix = np.array([[1.0/scale,0.0,0.0],[0.0,1.0/scale,0.0],[0.0,0.0,1.0]])

    return np.dot(scaling_matrix, homography_matrix)



def postprocess(feature_map, scale=2):
    #Scale = 2 by default, assuming videos are inputed in 1280×720
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x,y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    return x, y