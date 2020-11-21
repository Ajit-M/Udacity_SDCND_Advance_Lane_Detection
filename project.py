import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
import pickle as pk
import glob 

#%matplotlib inline

path = "/media/luffy/New Volume/Programming/python-p/Udacity_SDND/Term 1/Computer_Vision/CarND-Advanced-Lane-Lines/"

fname = glob.glob(path+"/camera_cal/*.jpg")

fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2)

counter = 0

#Preparing the array for ObjectPoints and the corresponding pixel location (ImagePoints) array
ObjPoints = np.zeros((9*6,3), np.float32)
ObjPoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Empty array of the ImgPoints and ObjPoints
ObjP = []
ImgP = []

# Setting the termination criteria for the cornerSubPix method
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Accessing all the images using glob module
for image in fname:
    input_image = mpimg.imread(image)

    gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray_image, (9,6), None)
    #print(ret)
    if(ret == True):
        counter = counter + 1
        ObjP.append(ObjPoints)
        #Finding more Accurate Corner points in image
        corners_accurate = cv.cornerSubPix(gray_image, corners, (11,11), (-1,-1), criteria)

        #print(dir(corners_accurate))
        # Appending the image points to the ImgP
        ImgP.append(corners_accurate)

        #Drawing the corners and Displaying them
        #cv.drawChessboardCorners(input_image, (9,6), corners_accurate, ret)
        #plt.imshow(input_image)
        #plt.show()

in_image = mpimg.imread(fname[3])

# Calibrating the camera using the Image Points and Object Points
ret, cam_mtx, dist, rvec, tvec = cv.calibrateCamera(ObjP, ImgP, in_image.shape[:-1], None, None)

#print(cam_mtx)
#print(ret)


# Getting Refining Camera Matrix
h,w = in_image.shape[:-1]
newCamMtx, roi = cv.getOptimalNewCameraMatrix(cam_mtx,dist, (w,h), 1, (w,h))


# Undistroting the image
undist = cv.undistort(in_image, cam_mtx, dist, newCamMtx)


print(counter)

#Cropping the image using ROI, got earlier cv.getOptimalNewCameraMatrix method
x,y,w,h = roi
undist = undist[y:y+h, x:x+w]
ax1.imshow(in_image)
ax2.imshow(undist)
plt.show()
        


