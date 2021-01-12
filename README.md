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

