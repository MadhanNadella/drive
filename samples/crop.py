import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def transform(pos):
	pts=[]
	n=len(pos)
	for i in range(n):
		pts.append(list(pos[i][0]))
	#print pts
	sums={}
	diffs={}
	for i in pts:
		x=i[0]
		y=i[1]
		sum=x+y
		diff=y-x
		sums[sum]=i
		diffs[diff]=i
	sums=sorted(sums.items())
	diffs=sorted(diffs.items())
	n=len(sums)
	rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
	#	   top-left   top-right   bottom-left   bottom-right
	
	h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)		#height of left side
	h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)		#height of right side
	h=max(h1,h2)
	
	w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)		#width of upper side
	w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)		#width of lower side
	w=max(w1,w2)
	
	#print '#',rect
	return int(w),int(h),rect
def sizee(arr):
    c=0
    pts=np.zeros(8)
    for i in np.ravel(arr):
        pts[c]=int(i/r)
        c+=1
    return np.float32([[pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],[pts[6],pts[7]]])
img=cv2.imread("pic47.jpg")
r=500.0 / img.shape[1]
dim=(500, int(img.shape[0] * r))
img1=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(11,11),0)
edge=cv2.Canny(gray,100,200)

contours,_=cv2.findContours(edge.copy(),1,1)
n=len(contours)
max_area=0
pos=0
for i in contours:
	area=cv2.contourArea(i)
	if area>max_area:
		max_area=area
		pos=i
peri=cv2.arcLength(pos,True)
approx=cv2.approxPolyDP(pos,0.02*peri,True)

size=img.shape
x=img.shape[1]
y=img.shape[0]
#cv2.imwrite("pic27re.jpg",img)
w,h,arr=transform(approx)

pts1=np.float32(arr)
pts2=np.float32([[0,0],[x,0],[0,y],[x,y]])

if cv2.contourArea(pos)<(0.25*x*y):
    pts1=pts2
else:
    pts1=sizee(pts1)
#pts1=np.float32(arr)
#pts2=sizee(pts2)

M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(x,y))
#cv2.imshow('window',dst)
#image=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#image=cv2.adaptiveThreshold(image,255,1,0,11,2)
#image = cv2.resize(image,(w,h),interpolation = cv2.INTER_AREA)
cv2.imwrite("pic47c.jpg",dst)


cv2.imwrite("pic47e.jpg",edge)
#cv2.imshow('img',edge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
