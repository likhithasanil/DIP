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
![Capture](https://user-images.githubusercontent.com/72268045/104289045-a7d29200-54de-11eb-8c28-4808f97deb66.PNG)


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


