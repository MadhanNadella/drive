import cv2
import numpy as np

large = cv2.imread('/home/madhan/EE04_Tess/newspaper/news1.png')
rgb = cv2.pyrDown(large)

cv2.imshow('rgb',rgb)

small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

cv2.imshow('small',small)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
#kernel = np.ones((11,11), np.uint8)
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

cv2.imshow("grad",grad)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("bw",bw)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

cv2.imshow('conn', connected)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#For opencv 3+ comment the previous line and uncomment the following line
#_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        #cv2.rectangle(large,(int()))

cv2.imshow("mask",mask)
cv2.imshow('rects', rgb)
imS = cv2.resize(rgb, (540, 960))
#cv2.imwrite()
cv2.imshow('rect', imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("/home/madhan/EE04_Tess/newspaper/news1a.png",rgb)
