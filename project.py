import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
import pickle as pk
import glob 

#%matplotlib inline
# Reading Camera Calibration Constants


#fig,  axes = plt.subplots(nrows= 8, ncols = 2, sharex=True, sharey=True)


with open('camera_calibration_constants.txt', 'rb') as fh:
    data = pk.load(fh)
        
ret, mtx, newCamMtx, dist, rvec, tvec = data

# Creating Image Pipeline

# Reading Images for the "test_images" folder

def read_images(dir_name):
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


def image_pipeline(fname):

    for x in range(0, len(fname)):

        input_image = mpimg.imread(fname[x])

        # Undistorting the image
        input_image = cv.undistort(input_image, mtx, dist, newCamMtx)    
        
        image_shape = input_image.shape
        print(image_shape)

        hls_image = cv.cvtColor(input_image, cv.COLOR_RGB2HLS)

        # Detecting Edges (Canny Detector)
        edges = cv.Canny(hls_image[:,:,2], 180, 240)
 

        # Creating the mask for selecting the Region of interest and warping it for the bird's eye view
        mask = np.zeros_like(edges)


        # Homography - Four point Transformation (Warping)
        
        left_bottom = [150, image_shape[0]]
        left_top = [450,500]
        right_top = [850,500]
        right_bottom = [1200, image_shape[0]]    


        points = np.array([[left_bottom,left_top,right_top,right_bottom]], dtype = np.int32) # The fillPoly method requires points in clockwise direction
       
        desiredPoints = np.array([[[0, image_shape[0]], [0,0], [image_shape[1],0 ], [image_shape[1],image_shape[0]]]], dtype= np.int32)

        print(points.shape)
        print(desiredPoints.shape)    
    

        cv.fillPoly(mask, points, (255,255,255))
        edge_img_roi = cv.bitwise_and(edges,mask)
        
        # Getting Perspective Transformation Matrix
        transformationMatrix = cv.getPerspectiveTransform(np.float32(points), np.float32(desiredPoints))
        #print(transformationMatrix)
        
        transformedImage = cv.warpPerspective(edge_img_roi, transformationMatrix, (image_shape[1],image_shape[0]))


        #axes[x+1,1].imshow(input_image)
        #axes[x+1,2].imshow(edges)
        plt.imshow(edge_img_roi)
        plt.show()
        plt.imshow(transformedImage)
        plt.show()


images = read_images("test_images")
image_pipeline(images)    


