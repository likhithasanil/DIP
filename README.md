# DIP
Q1. Develop the program to read and write the image.




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

** output **

Q2) Develop the program to perform linear transformation on image.
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

** output **
![Capture1](https://user-images.githubusercontent.com/72268045/104289786-92aa3300-54df-11eb-82ba-d7e40d5d134e.PNG)

B) Resizing of image.
import cv2
import numpy as np
img = cv2.imread('p17.jpg')
(height, width) = img.shape[:2]
res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC)
cv2.imshow('result', res)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

** output **
![Captur](https://user-images.githubusercontent.com/72268045/104290241-21b74b00-54e0-11eb-89b1-8832f79f8247.PNG)


Q3) Develop a program to find the sum and mean of set of image
create n number of images and read from directory and perform the operation
import cv2
import os
path=("D:\Image")
imgs=[]
dirs=os.listdir(path)
for file in dirs:
fpath=path+"\\"+file
imgs.append(cv2.imread(fpath))
i=0
for im in imgs:
cv2.imshow(dirs[i],imgs[i])
i=i+1
cv2.waitKey(0)
cv2.imshow("mean",len(im)/im)
cv2.waitKey(0)
cv2.imshow("sum",len(im))
cv2.waitKey(0)
cv2.destroyAllWindows()

** output **
![Captu](https://user-images.githubusercontent.com/72268045/104291986-414f7300-54e2-11eb-84f6-2df2d4dc02d6.PNG)

