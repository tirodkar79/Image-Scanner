from cv2 import cv2
import numpy as np

"""
This function is taken from pyimagesearch.com 
In this function we find the four points of the document
It is done by taking the target contour and determining it's four corners 
"""
def mapp(h):
    h = h.reshape((4,2))
    # We create an array for corners.
    hnew = np.zeros((4,2), dtype = np.float32)

    # For top left and bottom right points we take the addition
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)] # The minimum value is given to top left corner
    hnew[2] = h[np.argmax(add)] # The maximum value is given to bottom right corner

    # For top right and bottom left points we take the difference
    diff = np.diff(h, axis = 1)
    hnew[1] = h[np.argmin(diff)] # The minimum value is given to top right corner
    hnew[3] = h[np.argmax(diff)] # The minimum value is given to bottom left corner

    # Return this corners which are retrived
    return hnew


image = cv2.imread(r'F:\Pratham\Python\cv projects\scanner app\test_img.jpg')

# Resize the image as you desire
image = cv2.resize(image,(1300,800))

# Save a copy of original image for later use
orig = image.copy()
# Convert the image into black and white for edge detection
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   
# For edge detection it is recommended to smoothen the image for better results
blur = cv2.GaussianBlur(gray,(5,5),0)   

# Fetch the edges of the image
# In this algorithm the edges with intensity greater than maximum threshold are fetched
# Also the values of intensity between minimum and maximum threshold are checked for any connection with edge having maximum intensity

edge = cv2.Canny(blur,50,80)
cv2.imshow("Canny Edge",edge)
cv2.waitKey(0)


# Find the contours in image 
# Here we used CHAIN_APPROX_SIMPLE to just store the main points of the contour instead  of storing the whole contour
contour, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour,key=cv2.contourArea, reverse=True)

# Each contour is checked and the contour which is closed and forming an square or rectangle are named as target
for c in contour:
    p = cv2.arcLength(c,True)
    # Not all contour's are perfect so little approixmation can be used
    approx = cv2.approxPolyDP(c,0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

approx = mapp(target)

# We define the range of the window (Top left, Top Right, Bottom Right, Bottom Left)
pts = np.float32([[0,0],[800,0],[800,800],[0,800]])

# We transform the contours according to our requirement using perspective transform
op = cv2.getPerspectiveTransform(approx, pts)

# We wrap the original image according to the target or transformed contour and we resize it as we desire.
dts = cv2.warpPerspective(orig, op, (800,800))

cv2.imshow("Main",dts)
cv2.waitKey(0)