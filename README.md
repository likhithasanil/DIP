# DIP
Q1. Develop a program to  display grayscale image using read and write the operation.

**Description**
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and white colors.
imread() : is used for reading an image.
imwrite(): is used to write an image in memory to disk.
imshow() :to display an image.
waitKey(): The function waits for specified milliseconds for any keyboard event. 
destroyAllWindows():function to close all the windows.
cv2. cvtColor() method is used to convert an image from one color space to another For color conversion, we use the function cv2. cvtColor(input_image, flag) where flag determines the type of conversion. For BGR Gray conversion we use the flags cv2.COLOR_BGR2GRAY 

**Program**
import cv2
import numpy as np
image = cv2.imread('p4.jpg')
cv2.imshow('Old', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.imwrite('sample.jpg',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**<br/>
![Capt](https://user-images.githubusercontent.com/72268045/104295083-08190200-54e6-11eb-9062-6354c9ef1b4e.PNG)


Q2) Develop the program to perform linear transformation on image.
**Description**
//Rotation
Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing.
cv2.getRotationMatrix2D :Perform the counter clockwise rotation
warpAffine()            :This function is the size of the output image, which should be in the form of (width, height). where width = number of columns, and height = number of rows.

**Program**
Rotation of the image:
import cv2
import numpy as np
img = cv2.imread('p17.jpg')
(rows, cols) = img.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1)
res = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('result', res)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**

![Capture1](https://user-images.githubusercontent.com/72268045/104289786-92aa3300-54df-11eb-82ba-d7e40d5d134e.PNG)



B) Resizing of image.
**Description**
//Scaling
Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications.
It helps in reducing the number of pixels from an image .
cv2.resize() method refers to the scaling of images. It helps in reducing the number of pixels from an image .

**Program**
import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)


** output **
![Captur](https://user-images.githubusercontent.com/72268045/104290241-21b74b00-54e0-11eb-89b1-8832f79f8247.PNG)


Q3) Develop a program to find the sum and mean of set of image
create n number of images and read from directory and perform the operation.
**Description**
TO add two images with the OpenCV function,
cv. add(), or simply by the numpy operation res = img1 + img2 we can add the images.
Mean()    :The function mean calculates the mean value M of array elements, independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image
append()  :This method in python adds a single item to the existing list.
listdir() : This method in python is used to get the list of all files and directories in the specified directory.

**Program**
import cv2
import os
path = 'C:\images'
imgs = []
files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of five pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of five pictures",meanImg)
cv2.waitKey(0)

** output **
![Capture](https://user-images.githubusercontent.com/72268045/104430787-feba8480-553b-11eb-81bc-e10ddfa20975.PNG)


Q4) Develop the program to convert color image to gray image and binary image.
**Description:
Threshold function : Now, to convert our image to black and white, we will apply the thresholding operation. 
To do it, we need to call the threshold function of the cv2 module.
For this we are going to apply the simplest thresholding approach, which is the binary thresholding.
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
Naturally, the threshold function allows us to specify  these 3 parameters. So, the first input of the function is the gray scale image to which we want to apply the operation.
As second input, it receives the value of the threshold. We will consider the value 127, which is in the middle of the scale of the values a pixel in gray scale can take (from 0 to 255).As third input, the function receives the user defined value 
Gray image   :Grayscale is a range of monochromatic shades from black to white. 
Binary image :A binary image is one that consists of pixels that can have one of exactly two colors, usually black and white.
cvtcolor     :cvtColor() method is used to convert an image from one color space to another.

**Program**
import cv2
originalImage = cv2.imread('p14.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh,blackAndWhiteImage )= cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imshow('Original image',originalImage)
cv2.imshow('Gray image', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

** output **
![1](https://user-images.githubusercontent.com/72268045/104327232-8b4f4d80-5510-11eb-9cc4-d71ce7667745.PNG)

Q5) Develop the program to change the image to different color spaces.
**Description:
color spaces :Color spaces are different types of color modes, used in image processing and signals and system for various purposes.
cvtcolor     :cvtColor() method is used to convert an image from one color space to another.
BGR2GRAY     :converts the original image to gray scale imagae.
HSV          : It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies                from 0-179, Saturation value varies from 0-255 and Value  varies from 0-255.
LAB          :converts the original image to lab which stands for L – Represents Lightness.A – Color component ranging from Green to Magenta.B – Color component ranging from                    Blue to Yellow.
HLS          :The HSL color space, also called HLS or HSI, stands for:Hue : the color type Ranges from 0 to 360° in most applications 
YUV          :Refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives               intensity information very differently from color information

**Program**
import cv2 img = cv2.imread("E:\\p14.jpg") 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS) 
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

** output **
![mn](https://user-images.githubusercontent.com/72268045/104328099-7cb56600-5511-11eb-952d-8f3e536aaac5.PNG)

Q6) program to create an image using 2D array
**Description :
2D array can be defined as an array of arrays. The 2D array is organized as matrices which can be represented as the collection of rows and columns. However, 2D arrays are created to implement a relational database look like data structure.
uint8 : Is an unsigned 8-bit integer that can represent valuese from 0-255.
pil   : It is the python imaginary library which provides the python interpretr with the image editing capablities 

**Program**
import cv2 as c
import numpy as np
from PIL import Image
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [255, 130, 0]
array[:,100:] = [0, 0, 255]
img = Image.fromarray(array)
img.save('image1.png')
img.show()
c.waitKey(0)

** output **
![mn](https://user-images.githubusercontent.com/72268045/104329358-d4080600-5512-11eb-9d19-6d7190aeabfc.PNG)


