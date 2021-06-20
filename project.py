from matplotlib.cm import ScalarMappable
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
import pickle as pk
import glob 
import sys
import moviepy.editor as mov

np.set_printoptions(threshold=sys.maxsize)
#%matplotlib inline
# Reading Camera Calibration Constants

"""
    Complete pipeline of the systems
        1. Taking input of the image
        2. Converting the data to HLS Format, reason being finding the edges of the lane lines is much effective in saturation part of the data, using the Canny Edge detector.
        3. Converting the image to the Grayscale
        4. Finding the edges
        5. Getting the ROI using a zeros mask
        6. Getting the birds using the four point homography 
        7. Finding the highest concentration of the white pixels in the bottom half of the image, this point will be starting point of the lane.
        8. distributing it to the left and right lane 
        9. Creating windows around the base of the lane 
        10. Giving the heights and margins to the windows
        11. Finding all the points those lie in that margin and appending to the list
        12. Fitting the curve those points 
        13. Drawing the lane line throught that second degree polynomial

"""

""" 
--> Work Remaining in this project
    - Using the opencv tuning method for getting all the thresholds

"""



#fig,  axes = plt.subplots(nrows= 8, ncols = 2, sharex=True, sharey=True)



def getCalibrationParams(fileName):
    with open(fileName, 'rb') as fh:
        data = pk.load(fh)
    return data



''' 
    Need to use the glob module, to get the images
'''

def readImages(dir_name):
    cwd = os.getcwd()
    dir_list = os.listdir(cwd)

    if dir_name in dir_list:
        print("Folder name provided exist, Proceding to Load the images")
    else:
        print("Folder name provided doesn't exsist")

    image_dir = cwd+"/" + dir_name
    fname = os.listdir(image_dir)
    for x in range(0,len(fname)):
        fname[x] = dir_name + "/" + fname[x] 
    return fname

def canny_thres(image, low_threshold = 100, high_threshold = 180):
    '''

        - This function takes in smoothed image as the input image,
            lower threshold and upper threshold values as the
            parameter for this function
        - Canny Edge detection filter takes the gradient of the image, so
            its necessary to provide what should be the lower threshold
            and upper threshold of the intensity of the image pixel that
            should be considered as the Edge pixel.
    '''
    # @) Shouldn't the GaussinaBlur come before the Canny Edge detection filter
    output_image = cv.GaussianBlur(image, (15,15), 0)
    output_image = cv.Canny(output_image, low_threshold, high_threshold)
    
    output_image[output_image!= 0] = 1
    return output_image




''' 
    Combining different threshold for better detection of the lanes (Sobel - magnitude, direction and   )
'''
def sobel_thres(in_img, orientation='x', lowerThreshold = 100 , upperThreshold = 180):

    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)
    
    # Reducing the Noise by applying the Gaussian Blur
    gray_img = cv.GaussianBlur(gray_img, (25,25) , 0)
    
    # Detecting Edges (Canny Detector)
    # edges = cv.Canny(in_img[:,:,2], lowerThreshold, upperThreshold)
    
    edges_binary = np.zeros_like(gray_img)
    
    if (orientation == 'x'):
        edgeX = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=25)
        abs_edgeX = np.absolute(edgeX)
        scaled_edgeX = (abs_edgeX/np.max(abs_edgeX))*255
        edges_binary[(scaled_edgeX >= lowerThreshold) & (scaled_edgeX <= upperThreshold)] = 1    
        
    if (orientation == 'y'):
        edgeY = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=25)
        abs_edgeY = np.absolute(edgeY)
        scaled_edgeY = (abs_edgeY/np.max(abs_edgeY))*255
        edges_binary[(scaled_edgeY >= lowerThreshold) & (scaled_edgeY <= upperThreshold)] = 1
    
    if (orientation == 'xy'):
        
        edgeX = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=25)
        abs_edgeX = np.absolute(edgeX)
        scaled_edgeX = (abs_edgeX/np.max(abs_edgeX))*255
        
        edgeY = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=25)
        abs_edgeY = np.absolute(edgeY)
        scaled_edgeY = (abs_edgeY/np.max(abs_edgeY))*255
        
        
        edges_binary[(scaled_edgeX >= lowerThreshold) & (scaled_edgeX <= upperThreshold) & (scaled_edgeY >= lowerThreshold) & (scaled_edgeY <= upperThreshold)] = 1

    return edges_binary

def mag_thres(in_img , lowerThreshold = 80, upperThreshold = 180):

    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)
    
    # Reducing the Noise by applying the Gaussian Blur
    gray_img = cv.GaussianBlur(gray_img, (25,25) , 1.3)

    
    edgeX = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=25)
    edgeY = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=25)
    
    grad_mag = np.sqrt((edgeX)**2 + (edgeY)**2)    
    grad_mag = np.absolute(grad_mag)
    
    scale_factor = np.max(grad_mag)/255
    
    # scaled_mag = np.zeros_like(gray_img)
    scaled_mag = (grad_mag/scale_factor).astype(np.uint8)
    
    edges_binary = np.zeros_like(scaled_mag)

    edges_binary[(scaled_mag >= lowerThreshold) & (scaled_mag <= upperThreshold)] = 1

    return edges_binary

def grad_dir(in_img, lowerThreshold = 0.7, upperThreshold = 1.3):

    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)

    # Reducing the Noise by applying the Gaussian Blur
    gray_img = cv.GaussianBlur(gray_img, (5,5) , 1.3)

    edgeX = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=25)
    edgeY = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=25)
    
    abs_edgeX = np.absolute(edgeX)
    abs_edgeY = np.absolute(edgeY)
    
    abs_grad = np.arctan2(abs_edgeY, abs_edgeX)
    
    
    edges_binary = np.zeros_like(abs_grad)
    edges_binary[(abs_grad >= lowerThreshold) & (abs_grad <= upperThreshold)] = 1
    
    return edges_binary


def color_lab_thres(in_img):
    
    #Converting the Input Image to gray scale.
    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)
    
    # Converting the input image to the LAB Color model
    lab_img = cv.cvtColor(in_img, cv.COLOR_RGB2LAB)
    
    #Splitting the color Channels
    l_channel = lab_img[:,:,0]
    b_channel = lab_img[:,:,2]
    
    # Using the input RGB image for detecting the yellow color lane
    l_binary = np.zeros_like(gray_img)
    b_binary = np.zeros_like(gray_img)
    
    # Thresholds for detecting yellow in blue channel
    b_thresh = (155, 200)
    
    # Thresholds for detecting the both yellow and white lane
    l_thresh = (180,255)
    
    # Thresholding the values yellow
    l_binary[(l_channel >= l_thresh[0])&(l_channel <= l_thresh[1])] = 1
    b_binary[(b_channel >= b_thresh[0])&(b_channel <= b_thresh[1])] = 1
    
    edges_combined = np.zeros_like(gray_img)
    edges_combined[(l_binary == 1) | (b_binary == 1)] = 1
    
    ''' 
    # Visualization Block for debugging and tuning the thresholds
    fig, ax = plt.subplots(2,3)
    
    ax[0,0].imshow(in_img)
    ax[0,0].set_title("Input Image")

    # Sperated Channels
    ax[0,1].imshow(l_channel, cmap="gray")
    ax[0,1].set_title("L Channel")

    ax[0,2].imshow(b_channel, cmap="gray")
    ax[0,2].set_title("B channel")

    ax[1,0].imshow(l_binary, cmap="gray")
    ax[1,0].set_title("L binary ")
    
    ax[1,1].imshow(b_binary, cmap="gray")
    ax[1,1].set_title("B binary")
    
    ax[1,2].imshow(edges_combined, cmap="gray")
    ax[1,2].set_title("Combined")

    plt.show() ''' 
    
    return edges_combined
    


def color_rgb_thres(in_img):
    
    # Splitting out the RGB Channels
    r_channel = in_img[:,:,0]
    g_channel = in_img[:,:,1] 
    b_channel = in_img[:,:,2]
    
    # Converting the Input Image to gray scale.
    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)

    R_thresh_Y = (170, 255)
    G_thresh_Y = (120, 180)
    
    R_thresh_W = (180, 255)
    G_thresh_W = (180, 255)
    B_thresh_W = (180, 255)
    
    # Using the input RGB image for detecting the yellow color lane
    r_binary_w = np.zeros_like(gray_img)
    g_binary_w = np.zeros_like(gray_img)
    b_binary_w = np.zeros_like(gray_img)
    
    # Using the input RGB image for detecting the yellow color lane
    r_binary_y = np.zeros_like(gray_img)
    g_binary_y = np.zeros_like(gray_img)
    
    # Thresholding the values white
    r_binary_w[(r_channel >= R_thresh_W[0])&(r_channel <= R_thresh_W[1])] = 1
    g_binary_w[(g_channel >= G_thresh_W[0])&(g_channel <= G_thresh_W[1])] = 1
    b_binary_w[(b_channel >= B_thresh_W[0])&(b_channel <= B_thresh_W[1])] = 1
    
    # Thresholding the values yellow
    r_binary_y[(r_channel >= R_thresh_Y[0])&(r_channel <= R_thresh_Y[1])] = 1
    g_binary_y[(g_channel >= G_thresh_Y[0])&(g_channel <= G_thresh_Y[1])] = 1
  
    edges_binary_rgb_w = np.zeros_like(gray_img)
    edges_binary_rgb_w[(r_binary_w == 1) & (g_binary_w == 1) & (b_binary_w == 1)] = 1
    
    edges_binary_rgb_y = np.zeros_like(gray_img)
    edges_binary_rgb_y[(r_binary_y == 1) & (g_binary_y == 1)] = 1
    
    edges_rgb_combined = np.zeros_like(gray_img)
    edges_rgb_combined[(edges_binary_rgb_w == 1) | (edges_binary_rgb_y == 1)] = 1
    
    
    ''' 
    # Visualization Block for debugging and tuning the thresholds
    fig, ax = plt.subplots(2,4)
    
    ax[0,0].imshow(in_img)
    ax[0,0].set_title("Input Image")

    # Sperated Channels
    ax[0,1].imshow(r_channel, cmap="gray")
    ax[0,1].set_title("R Channel")

    ax[0,2].imshow(g_channel, cmap="gray")
    ax[0,2].set_title("G channel")

    # Thresholded binaries
    ax[0,3].imshow(b_channel, cmap="gray")
    ax[0,3].set_title("B Channel")

    ax[1,0].imshow(edges_binary_rgb_w, cmap="gray")
    ax[1,0].set_title("White ")
    
    ax[1,1].imshow(edges_binary_rgb_y, cmap="gray")
    ax[1,1].set_title("Yellow")
    
    ax[1,2].imshow(edges_rgb_combined, cmap="gray")
    ax[1,2].set_title("Combined")

    plt.show() ''' 

    return edges_rgb_combined



def color_hls_thres(in_img):
    
    # Converting the Input Image to gray scale.
    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)

    # Converting the color space to HLS for more accurate detection of the yellow color lane
    hls_image = cv.cvtColor(in_img, cv.COLOR_RGB2HLS)
    
    # Using the S Channel to detect the yellow lane
    l_channel = hls_image[:,:,1]
    s_channel = hls_image[:,:,2]
    
    # For HLS color Thresholds for Yellow Color Lane
    S_thresh =  (80,255) 
    L_thresh = (180,255)
    
    # Creating the mask for the S thresholding
    s_binary = np.zeros_like(gray_img)
    l_binary = np.zeros_like(gray_img)
    
    s_binary[(s_channel >= S_thresh[0]) & (s_channel <= S_thresh[1])] = 1
    l_binary[(l_channel >= L_thresh[0]) & (l_channel <= L_thresh[1])] = 1

    edges_binary_hls = np.zeros_like(gray_img)
    edges_binary_hls[(s_binary == 1) | (l_binary == 1)] = 1

    ''' 
    fig, ax = plt.subplots(2,3)
    
    ax[0,0].imshow(in_img)
    ax[0,0].set_title("Input Image")
    
    ax[0,1].imshow(s_channel, cmap="gray")
    ax[0,1].set_title("S Channel Seperated")
    
    ax[0,2].imshow(s_binary, cmap="gray")
    ax[0,2].set_title("S Channel Thresholded") 
    
    ax[1,0].imshow(edges_binary_hls, cmap="gray")
    ax[1,0].set_title("Combined")
    
    ax[1,1].imshow(l_channel, cmap="gray")
    ax[1,1].set_title("L Channel Seperated")
    
    ax[1,2].imshow(l_binary, cmap="gray")
    ax[1,2].set_title("L Channel Thresholded") 
        
    plt.show() '''
    
    return edges_binary_hls



def color_hsv_thres(in_img):
    
        
    # Converting the Input Image to gray scale.
    gray_img = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)

    # Converting the color space to HLS for more accurate detection of the yellow color lane
    hsv_image = cv.cvtColor(in_img, cv.COLOR_RGB2HSV)
    
    
    # Splitting out the H channel, S channel, and V Channel 
    h_channel = hsv_image[:,:,0]  
    s_channel = hsv_image[:,:,1]
    v_channel = hsv_image[:,:,2]
  
    # For HLS color Thresholds for Yellow Color Lane
    S_thresh =  (80,255) 
    V_thresh = (180,255)

    

    # Creating the mask for the value Channel thresholding  
    v_binary = np.zeros_like(gray_img)
    v_binary[(v_channel >= V_thresh[0]) & (v_channel <= V_thresh[1])] =1

    # Creating the mask for the S Channel Thresholding
    s_binary = np.zeros_like(gray_img)
    s_binary[(s_channel >= S_thresh[0]) & (s_channel <= S_thresh[1])] = 1
        
    # Combining the masks of V and S Channel thresholding
    edges_binary_hsv = np.zeros_like(gray_img)
    edges_binary_hsv[(v_binary == 1) | (s_binary == 1) ] = 1
    
    ''' 
    fig, ax = plt.subplots(2,3)
    
    ax[0,0].imshow(in_img)
    ax[0,0].set_title("Input Image")
    
    ax[0,1].imshow(s_channel, cmap="gray")
    ax[0,1].set_title("S Channel Seperated")
    
    ax[0,2].imshow(s_binary, cmap="gray")
    ax[0,2].set_title("S Channel Thresholded") 
    
    ax[1,0].imshow(v_channel, cmap="gray")
    ax[1,0].set_title("V channel")
    
    ax[1,1].imshow(v_binary, cmap="gray")
    ax[1,1].set_title("V Channel thresh")
    
    ax[1,2].imshow(edges_binary_hsv, cmap="gray")
    ax[1,2].set_title("Combined") 
        
    plt.show() '''
    
    return edges_binary_hsv



def roi(edge_binary, points):
    # Creating the mask for selecting the Region of interest and warping it for the bird's eye view
    mask = np.zeros_like(edge_binary)    
    cv.fillPoly(mask, points, (255,255,255))
    edge_img_roi = cv.bitwise_and(edge_binary,mask)
    return edge_img_roi



''' 

'''
def homographyTransform(edge_img_roi, image_shape, points):
    
    desiredPoints = np.array([[[0, image_shape[0]], [0,0], [image_shape[1],0 ], [image_shape[1],image_shape[0]]]], dtype= np.int32)
    
    
    # Getting Perspective Transformation Matrix
    transformationMatrix = cv.getPerspectiveTransform(np.float32(points), np.float32(desiredPoints))

    transformedImage = cv.warpPerspective(edge_img_roi, transformationMatrix, (image_shape[1],image_shape[0]))
    return transformedImage






def inverseHomographyTransform(edge_img_roi, image_shape, points):
    
    desiredPoints = np.array([[[0, image_shape[0]], [0,0], [image_shape[1],0 ], [image_shape[1],image_shape[0]]]], dtype= np.int32)
    
    
    # Getting Perspective Transformation Matrix
    transformationMatrix = cv.getPerspectiveTransform(np.float32(desiredPoints), np.float32(points))
    #print(transformationMatrix)
    
    transformedImage = cv.warpPerspective(edge_img_roi, transformationMatrix, (image_shape[1],image_shape[0]))
    return transformedImage






def fitPolynomial(leftx, lefty, rightx, righty, input_image):
        
    # Fitting the Polynomial through the Lanes
    
    left_poly_fit = np.polyfit(lefty, leftx,2) # Here the order are reversed because we are predicting the value of x and we know y because we generated it using linspace
    right_poly_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, int(input_image.shape[0])-1, int(input_image.shape[0]))
    # print(left_fit, "\n", right_fit)
    return left_poly_fit, right_poly_fit




def slidingWindow(input_image, transformedImage, points):    

    # Finding Histogram Peaks in the Bottom part of the images
    bottom_image = transformedImage[transformedImage.shape[0]//2:,:]

    histogram = np.sum(bottom_image, axis=0)

    # Splitting the image in two halves
    midpoint = np.int(histogram.shape[0]//2)
    leftx_lane_base = np.argmax(histogram[:midpoint])
    rightx_lane_base = midpoint + np.argmax(histogram[midpoint:])

    # print(leftx_lane_base)
    # print(midpoint)
    # print(rightx_lane_base)

    # Setting up Hyperparameters for sliding window method
    nwindows = 7 # Number of windows in the image
    margin = 100 # Margin the windows to considered on either side
    minpix = 200 # Minimum pixels to recenter the window 
    window_height = np.int(transformedImage.shape[0]//nwindows)

    # Identifying the pixels that are activated (i.e. Non Zero pixels) in the window
    nonzero = transformedImage.nonzero()
    # print(nonzero)

    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    # Current position of the window, starting from the bottom of the image and moving up
    leftx_current = leftx_lane_base
    rightx_current = rightx_lane_base

    #Empty list of the indices for the left lane and right lane
    left_lane_indices = np.array([])
    right_lane_indices = np.array([])
    
        
    output_image = np.int8(np.dstack((transformedImage, transformedImage, transformedImage))*255)

    # Going through all the windows
    for window in range(nwindows):
        # Identifying the window boundaries for both the left and right lane
        win_high_y = transformedImage.shape[0] - (window)*(window_height)
        win_low_y = transformedImage.shape[0] - (window+1)*window_height
        win_left_x_low = leftx_current - margin
        win_left_x_high = leftx_current + margin
        win_right_x_low = rightx_current - margin
        win_right_x_high = rightx_current + margin

        # Drawing the rectangles of windows
        cv.rectangle(output_image,(win_left_x_high, win_high_y),(win_left_x_low, win_low_y),(0,255,0),2)
        cv.rectangle(output_image,(win_right_x_high, win_high_y),(win_right_x_low, win_low_y),(0,255,0),3)


        # Identifying the pixels that are useful
        good_left_indices = ((nonzero_y >= win_low_y) & (nonzero_y < win_high_y) & ( nonzero_x >= win_left_x_low) & (nonzero_x < win_left_x_high)).nonzero()[0]

        good_right_indices = ((nonzero_y >= win_low_y) & (nonzero_y < win_high_y) & ( nonzero_x >= win_right_x_low) & (nonzero_x < win_right_x_high)).nonzero()[0]

        # Recentering the window based upon the minimum number of pixels found
        if len(good_left_indices) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_indices]))
        if len(good_right_indices) > minpix:    
            rightx_current = np.int(np.mean(nonzero_x[good_right_indices]))

        # print(good_left_indices.shape)udac
        left_lane_indices = np.append(left_lane_indices, good_left_indices)
        right_lane_indices = np.append(right_lane_indices, good_right_indices)
        
        
        
        leftx = nonzero_x[left_lane_indices.astype(int)]
        lefty = nonzero_y[left_lane_indices.astype(int)] 
        rightx = nonzero_x[right_lane_indices.astype(int)]
        righty = nonzero_y[right_lane_indices.astype(int)]
        
    if (len(leftx) != 0 and len(rightx) != 0):
        #Fitting the Second degree Polynomial
         left_poly_fit, right_poly_fit = fitPolynomial(leftx, lefty, rightx, righty, input_image)
         left_curvature, right_curvature = lane_curvature(input_image,left_poly_fit, right_poly_fit)
         offset = car_offset(transformedImage, left_poly_fit, right_poly_fit)
         output_image = visualization(input_image, left_poly_fit, right_poly_fit, points, left_curvature, right_curvature, offset)
    else:
        print("Length of Left lane X values", len(leftx), "and length of Right Lane X values", len(rightx))
        pass
        
    return output_image, left_curvature, right_curvature, offset, left_poly_fit, right_poly_fit
    
 

def sliding_window_priori(left_prev_fit, right_prev_fit, input_image, transformedImage, points):

    """ 
    Input will be two arrays - left_fit and right_fit and binary image 
    
    """
    
    nonzero = transformedImage.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])
    margin = 100
    left_lane_indices = ((nonzero_x > (left_prev_fit[0]*(nonzero_y**2) + left_prev_fit[1]*nonzero_y + left_prev_fit[2]- margin)) & \
                        (nonzero_x < (left_prev_fit[0]*(nonzero_y**2) + left_prev_fit[1]*nonzero_y + left_prev_fit[2] + margin )))
    
    right_lane_indices = (( nonzero_x > (right_prev_fit[0]*nonzero_y**2 + right_prev_fit[1]* nonzero_y + right_prev_fit[2] - margin ))& \
                         (nonzero_x < (right_prev_fit[0]*nonzero_y**2 + right_prev_fit[1]* nonzero_y + right_prev_fit[2] + margin )))
    
    # print(left_lane_indices)
    
    # Getting the left and right lane pixels 
    leftx = nonzero_x[left_lane_indices]
    lefty = nonzero_y[left_lane_indices]
    rightx =  nonzero_x[right_lane_indices]
    righty =  nonzero_y[right_lane_indices]
    
    
    if (len(leftx) <= 3300 or len(rightx) <= 3300):
        # Instead of recomputing the polynomial fit, use the previous fit since the points are less (Less probability)
        
        left_poly_fit = left_prev_fit
        right_poly_fit = right_prev_fit
        
        print("Previous Length of Left lane X values", len(leftx), "and length of Right Lane X values", len(rightx))
                    
        left_curvature, right_curvature = lane_curvature(input_image,left_poly_fit, right_poly_fit)
        
        offset = car_offset(transformedImage, left_poly_fit, right_poly_fit)
        
        output_image = visualization(input_image, left_poly_fit, right_poly_fit, points, left_curvature, right_curvature, offset)
    else:
        # Here the probability of the lane is higher since the number of the detected points are higher than the lower bound
        
        #Fitting the Second degree Polynomial
        print("Length of Left lane X values", len(leftx), "and length of Right Lane X values", len(rightx))
        
        left_poly_fit, right_poly_fit = fitPolynomial(leftx, lefty, rightx, righty, input_image)
        
        left_curvature, right_curvature = lane_curvature(input_image,left_poly_fit, right_poly_fit)
        
        offset = car_offset(transformedImage, leftx, rightx)
        
        output_image = visualization(input_image, left_poly_fit, right_poly_fit, points, left_curvature, right_curvature, offset)
        
    
    return output_image, left_curvature, right_curvature, offset, left_poly_fit, right_poly_fit


def lane_curvature(input_image,left_fit, right_fit):
    
    
    """ 
    Consider the assumptions made in the exercise before, converting the pixel values to the real world

    Assumptinos
        1. Road spans 30m long 
        2. Width of the road is 12  
    
    """
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension 
    
    ## Need to generate the x,y points using the poly fit convert them according the meter to pixel ratio and then compute curvature
    
    ploty = np.linspace(0, int(input_image.shape[0])-1, int(input_image.shape[0]))
    
    left_x_points = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_x_points = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Converting the pixel distance to the real world
    ploty = ploty * ym_per_pix
    
    left_fit = np.polyfit(left_x_points*ym_per_pix, left_x_points*xm_per_pix,2 ) 
    right_fit = np.polyfit(right_x_points*ym_per_pix, right_x_points*xm_per_pix,2 ) 
    
    # Finding the curvature of the lanes using the formula
    left_curvature = ( (1+(2*left_fit[0])**2)**(3/2) / abs(2*left_fit[0]) )
    right_curvature = ( (1+(2*right_fit[0])**2)**(3/2) / abs(2*right_fit[0]) )
    
    return left_curvature, right_curvature


def car_offset(transformed_image,left_poly_fit, right_poly_fit):
    
    xm_per_pix = 3.7/700 
    mid_point = transformed_image.shape[-1]//2

    ploty = np.linspace(0, int(transformed_image.shape[0])-1, int(transformed_image.shape[0]))

    
    left_x_points = left_poly_fit[0]*ploty**2 + left_poly_fit[1]*ploty + left_poly_fit[2]
    right_x_points = right_poly_fit[0]*ploty**2 + right_poly_fit[1]*ploty + right_poly_fit[2]
    
    # Need to understand this formula to find the car position or offset from the center
    car_position = (left_x_points[-1] + right_x_points[-1])/2
    
    offsetx = (mid_point - car_position) * xm_per_pix
    
    return offsetx
    
    

# Image Pipeline

def undistrot(input_image, data):
    
    ret, mtx, newCamMtx, dist, rvec, tvec = data
    # Undistorting the image
    input_image = cv.undistort(input_image, mtx, dist, newCamMtx)  
    return input_image


''' 
    This function is just for the displaying the results.

'''
def visualization(input_image, left_fit_x, right_fit_x, points, left_curvature, right_curvature, offset):

    input_image_shape = input_image.shape

    zeros_mask = np.zeros_like(input_image)

    ploty = np.linspace(0, int(input_image.shape[0])-1, int(input_image.shape[0]))
    
    left_x_points = left_fit_x[0]*ploty**2 + left_fit_x[1]*ploty + left_fit_x[2]
    right_x_points = right_fit_x[0]*ploty**2 + right_fit_x[1]*ploty + right_fit_x[2]
    
    # Drawing the lines on to the image
    drawPoints_left = (np.asarray([left_x_points, ploty]).T) 
    drawPoints_right = (np.asarray([right_x_points, ploty]).T)


    # Recasting the points for a suitbale shape
    pts = np.hstack((drawPoints_left, drawPoints_right)) # How does the hstack works but not the vstack method? Am I missing something?

    pts = pts.reshape(-1,2)
    
    # Drawing the Lane lines, area between the lane lines
    cv.fillPoly(zeros_mask, np.int_([pts]), (0,255,0))  
    cv.polylines(zeros_mask, np.int32([drawPoints_left]), False, (0,0,255), thickness=25 )
    cv.polylines(zeros_mask, np.int32([drawPoints_right]), False, (255,0,0), thickness=25 )

    # Computing the inverse Homography 
    zeros_mask = inverseHomographyTransform(zeros_mask, input_image_shape, points)

    # output_image = input_image + zeros_mask  
    output_image = cv.addWeighted(input_image, 1, zeros_mask, 0.7, 0.0)

    # Adding the text information to the frame 
    cv.putText(output_image, "The Left Lane Curvature is : " + str(left_curvature) + " m", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 5)
        
    cv.putText(output_image, "The Right Lane Curvature is : " + str(right_curvature) + " m", (100,160), cv.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 5)

    cv.putText(output_image, "The Car Offset is : " + str(offset) + " m", (100,220), cv.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 5)

    return output_image
    


def image_pipeline(fname):

    for x in range(0, len(fname)):
        
        # Get the calibarantion Parameters saved in a file 
        cal_data = getCalibrationParams('camera_calibration_constants.txt')
        
        # Read the Image
        input_image = cv.imread(fname[x])

        if input_image.all() == None:
            print("Image Reading Error")
            
        else:    
            
            # Remove the Distortion in the image
            undistorted_image = undistrot(input_image,cal_data)
            
            image_shape = undistorted_image.shape
            
            # Convert the color scale to RGB
            rgb_image = cv.cvtColor(undistorted_image, cv.COLOR_BGR2RGB)
            # rgb_image = undistorted_image
            gray_img = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
            
            plt.imshow(rgb_image)
            plt.title('Input Images')
            plt.show()
            
            # Combining binary filter and binary image from color space 
            
            
            # Finding the Region of Interest and Masking it
            
            left_bottom = [0, image_shape[0]]
            left_top = [480,465]
            right_top = [850,465]
            right_bottom = [1250, image_shape[0]] 
            
            points = np.array([[left_bottom,left_top,right_top,right_bottom]], dtype =np.int32)
            #print(points.shape)
            roiEdgeImage = roi(rgb_image, points)
            
            # Homography Transformation        
            transformedImage = homographyTransform(roiEdgeImage, image_shape, points)
            
            binary_transformedImage = homographyTransform(rgb_image, image_shape, points)
            
            
            
            # Edge Detection
            binary_sobel = sobel_thres(binary_transformedImage)
            binary_mag = mag_thres(binary_transformedImage)
            binary_gradDir = grad_dir(binary_transformedImage)
            binary_canny = canny_thres(binary_transformedImage)
            
            print(np.max(binary_canny))
            
              
            plt.imshow(binary_sobel, cmap="gray")
            plt.title("Sobel binary")
            plt.show()
            
            plt.imshow(binary_mag, cmap="gray")
            plt.title("Magnitude Thresholding binary")
            plt.show()
            
            plt.imshow(binary_gradDir, cmap="gray")
            plt.title("Gradient Direction binary")
            plt.show()
            
            plt.imshow(binary_canny, cmap="gray")
            plt.title("Canny")
            plt.show()
            
                        
                            
            binary_filter = np.zeros_like(gray_img)
            
            binary_filter[(binary_sobel == 1) & (binary_mag == 1) & (binary_canny == 1) & (binary_gradDir == 1)]  = 1
            
        
            # Color Thresholding for edge detection
            color_binary_hsv = color_hsv_thres(transformedImage)
            color_binary_hls = color_hls_thres(transformedImage)            
            # color_binary_rgb = color_rgb_thres(transformedImage)
            color_binary_lab = color_lab_thres(transformedImage)
            
            
            color_binary_hsv = cv.bitwise_or(color_binary_hsv, binary_filter)
            color_binary_hls = cv.bitwise_or(color_binary_hls, binary_filter)            
            # color_binary_rgb = cv.bitwise_or(color_binary_rgb, binary_filter)
            color_binary_lab = cv.bitwise_or(color_binary_lab, binary_filter)
            
            color_binary_final = np.zeros_like(gray_img)

            color_binary_final[(color_binary_hsv == 1) & (color_binary_hls == 1) &(color_binary_lab == 1)] = 1

            binary_image = color_binary_final
            
            
            
            # color_binary_final[(color_binary_hsv == 1) | (color_binary_hls == 1) | (color_binary_rgb == 1) | (color_binary_lab == 1)] = 1
            
           
            
            plt.imshow(color_binary_final, cmap="gray")
            plt.title("Color Binary final")
            plt.show()
            
            # binary_image = np.zeros_like(gray_img)
            
            # binary_image[(binary_filter == 1) | (color_binary_final == 1)] = 1
            
            # plt.imshow(binary_image, cmap="gray")
            # plt.title("Combined Color and gradient")
            # plt.show()
            
            
            # Implementing Sliding Window Algorithm
            out_image, left_curvature, right_curvature, offset, left_fit, right_fit = slidingWindow(input_image, binary_image, points)
            
            plt.imshow(out_image)
            plt.show()


''' 
    1. Adding the flag to use prior information 

'''
def video_pipeline(videos):
     # Get the calibarantion Parameters saved in a file 
    cal_data = getCalibrationParams('camera_calibration_constants.txt')
        
    for x in range(0, len(videos)):
        flag = False   
        capture = mov.VideoFileClip(videos[x]) 

        for frames in capture.iter_frames():
            # Read the Image
            input_image = frames

            if input_image.all() == None:
                print("Image Reading Error")
                
            else:    
                # Remove the Distortion in the image
                undistorted_image = undistrot(input_image,cal_data)
                
                image_shape = undistorted_image.shape
                
                # Convert the color scale to RGB
                # rgb_image = cv.cvtColor(undistorted_image, cv.COLOR_BGR2RGB)
                rgb_image = undistorted_image
                gray_img = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
                
               
               
                # Finding the Region of Interest and Masking it
                
                left_bottom = [0, image_shape[0]]
                left_top = [490,490]
                right_top = [850,490]
                right_bottom = [1250, image_shape[0]] 
                
                points = np.array([[left_bottom,left_top,right_top,right_bottom]], dtype =np.int32)
                #print(points.shape)
                roiEdgeImage = roi(rgb_image, points)
                
                # Homography Transformation        
                transformedImage = homographyTransform(roiEdgeImage, image_shape, points)
                

                # Edge Detection
                binary_sobel = sobel_thres(transformedImage)
                binary_mag = mag_thres(transformedImage)
                binary_gradDir = grad_dir(transformedImage)
                binary_canny = canny_thres(transformedImage)
                               
                            
                binary_filter = np.zeros_like(gray_img)
                
                binary_filter[(binary_sobel == 1) & (binary_mag == 1) & (binary_canny == 1) & (binary_gradDir == 1)]  = 1
                
                               
            
                # Color Thresholding for edge detection
                color_binary_hsv = color_hsv_thres(transformedImage)
                color_binary_hls = color_hls_thres(transformedImage)            
                color_binary_rgb = color_rgb_thres(transformedImage)
                color_binary_lab = color_lab_thres(transformedImage)
                
                
                # color_binary_hsv = cv.bitwise_or(color_binary_hsv, binary_filter)
                # color_binary_hls = cv.bitwise_or(color_binary_hls, binary_filter)            
                # # color_binary_rgb = cv.bitwise_or(color_binary_rgb, binary_filter)
                # color_binary_lab = cv.bitwise_or(color_binary_lab, binary_filter)
                
                color_binary_final = np.zeros_like(gray_img)



                color_binary_final[(color_binary_hsv == 1) & (color_binary_hls == 1) & (color_binary_lab == 1)] = 1

                # binary_image = color_binary_final
            
                binary_image = np.zeros_like(gray_img)
            
                binary_image[(binary_filter == 1) | (color_binary_final == 1)] = 1    
                    
                if(flag == True):
                    # Implementing Sliding Window Algorithm
                    out_image, left_curvature, right_curvature, offset , left_fit, right_fit = sliding_window_priori(left_fit, right_fit, input_image, binary_image, points)
                    
                else:    
                    # Implementing Sliding Window Algorithm
                    out_image, left_curvature, right_curvature, offset, left_fit, right_fit = slidingWindow(input_image, binary_image, points)
                    flag = True
                # plt.imshow(out_image)
                # plt.show()
                
                out_image = cv.cvtColor(out_image, cv.COLOR_RGB2BGR)
                
                # binary_image = np.stack((binary_image,binary_image,binary_image))
                binary_image = (inverseHomographyTransform(binary_image,image_shape, points))*255
                
                color_binary_final = color_binary_final *255
                binary_sobel = binary_sobel*255
                
                cv.imshow("video", binary_image)
                cv.imshow("video1", out_image)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
      



# images = readImages("image")

# image_pipeline(images)    

videos = readImages("test_video")

video_pipeline(videos)

cv.destroyAllWindows()
