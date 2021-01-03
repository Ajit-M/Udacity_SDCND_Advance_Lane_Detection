import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
import pickle as pk
import glob 
import sys

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

    1. Complete the current implementation of Udacity
    Converting the code into functions
        b. Multiple Plot (To be done last)
        c. Combine the road image with the Boxes
        
        d. Fine tuning the filter or adding if required to completely remove the noise from the middle of the road
        e. 

"""


#fig,  axes = plt.subplots(nrows= 8, ncols = 2, sharex=True, sharey=True)



def getCalibrationParams(fileName):
    with open(fileName, 'rb') as fh:
        data = pk.load(fh)
    return data


        

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


def edgeDetector(in_img, lowerThreshold = 180, upperThreshold = 240):
    
    # Detecting Edges (Canny Detector)
    edges = cv.Canny(in_img[:,:,2], lowerThreshold, upperThreshold)
    
    edges_binary = np.zeros_like(edges)
    edges_binary[(edges >= 180)&(edges <= 255)] = 1
    return edges_binary


def roi(edge_binary, points):
    # Creating the mask for selecting the Region of interest and warping it for the bird's eye view
    mask = np.zeros_like(edge_binary)    
    cv.fillPoly(mask, points, (255,255,255))
    edge_img_roi = cv.bitwise_and(edge_binary,mask)
    return edge_img_roi
    
def homographyTransform(edge_img_roi, image_shape, points):
    
    desiredPoints = np.array([[[0, image_shape[0]], [0,0], [image_shape[1],0 ], [image_shape[1],image_shape[0]]]], dtype= np.int32)
    
    
    # Getting Perspective Transformation Matrix
    transformationMatrix = cv.getPerspectiveTransform(np.float32(points), np.float32(desiredPoints))
    #print(transformationMatrix)
    
    transformedImage = cv.warpPerspective(edge_img_roi, transformationMatrix, (image_shape[1],image_shape[0]))
    return transformedImage


def slidingWindow(transformedImage):    

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
    nwindows = 16 # Number of windows in the image
    margin = 100 # Margin the windows to considered on either side
    minpix = 50 # Minimum pixels to recenter the window 
    window_height = np.int(transformedImage.shape[0]//nwindows)

    # Identifying the pixels that are activated (i.e. Non Zero pixels) in the window
    nonzero = transformedImage.nonzero()

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

        # plt.imshow(transformedImage)
        # plt.show()

        # Visualization bounding box data
        
        
        print(transformedImage.shape)
        # Drawing the rectangles of windows
        cv.rectangle(output_image,(win_left_x_high, win_high_y),(win_left_x_low, win_low_y),(0,255,0),2)
        cv.rectangle(output_image,(win_right_x_high, win_high_y),(win_right_x_low, win_low_y),(0,255,0),3)
        
        
        

        # plt.imshow(transformedImage, cmap="gray")
        # plt.show()


        # Identifying the pixels that are useful
        good_left_indices = ((nonzero_y >= win_low_y) & (nonzero_y < win_high_y) & ( nonzero_x >= win_left_x_low) & (nonzero_x < win_left_x_high)).nonzero()[0]

        good_right_indices = ((nonzero_y >= win_low_y) & (nonzero_y < win_high_y) & ( nonzero_x >= win_right_x_low) & (nonzero_x < win_right_x_high)).nonzero()[0]

        # print(good_left_indices)
        # print(good_right_indices)

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

    return leftx, lefty, rightx, righty, output_image
    
 

def fitPolynomial(leftx, lefty, rightx, righty, input_image, output_image):
    
    # Fitting the Polynomial through the Lanes
    
    left_fit = np.polyfit(leftx, lefty,2)
    right_fit = np.polyfit(rightx, righty, 2)
    
    ploty = np.linspace(0, int(input_image.shape[0])-1, int(input_image.shape[0]))
    left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # print(left_fit_x)
    # print(right_fit_x)
    
        ## Visualization ##
    # Colors in the left and right lane regions
    output_image[lefty, leftx] = [255, 0, 0]
    output_image[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fit_x, ploty, color='red')
    plt.plot(right_fit_x, ploty, color='yellow')
    return output_image



# Creating Image Pipeline

# Reading Images for the "test_images" folder


def undistrot(input_image, data):
    
    ret, mtx, newCamMtx, dist, rvec, tvec = data
    # Undistorting the image
    input_image = cv.undistort(input_image, mtx, dist, newCamMtx)  
    return input_image


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
        
            # Convert the color scale to HLS for threshloding and Edge Detection
            hls_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2HLS)
            
            plt.imshow(rgb_image)
            plt.show()
    
            # Edge Detection
            binary_image = edgeDetector(hls_image)
            
            # plt.imshow(binary_image, cmap="gray")
            # plt.show()
            
            # Finding the Region of Interest and Masking it
            
            left_bottom = [150, image_shape[0]]
            left_top = [450,500]
            right_top = [850,500]
            right_bottom = [1200, image_shape[0]] 
            
            points = np.array([[left_bottom,left_top,right_top,right_bottom]], dtype =np.int32)
            #print(points.shape)
            roiEdgeImage = roi(binary_image, points)
            
            # Homography Transformation        
            transformedImage = homographyTransform(roiEdgeImage, image_shape, points)
            
            # Implementing Sliding Window Algorithm
            leftx, lefty, rightx, righty, out_image = slidingWindow(transformedImage)
            
            
            
            plt.imshow(out_image)
            plt.show()


images = readImages("test_images")

image_pipeline(images)    


