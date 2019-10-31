import cv2
import perspective as per
import pytesseract as pyt
import binvalues as biv
import alphaToBraille as alp

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value


# Loading model
model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph1.pb',
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
cap = cv2.VideoCapture("test2.mp4")

i=0
taken = False
while(cap.isOpened() and not taken):
    _,image = cap.read()
    image_height, image_width, _= image.shape

    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()
    # print(output[0,0,:,:].shape)


    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .5:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            # print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
            if(class_id==84):
                center = ((box_x+box_width)/2, (box_y+box_height)/2 )
                if (center[0]< (image_width/2)-(0.1*image_width)):
                    print("Move Left")
                    i=0
                elif (center[0]> (image_width/2)+(0.1*image_width)):
                    print("Move Right")
                    i=0
                elif (center[1]< (image_height/2)-(0.1*image_height)):
                    print("Move Up")
                    i=0
                elif (center[1]> (image_height/2)+(0.1*image_height)):
                    print("Move Down")
                    i=0
                elif (((box_height - box_y)<0.2 * image_height) or ((box_width - box_x)<0.2 * image_width)):
                    print("Move Towards the Book")
                    i=0
                elif ( ((box_x < 0.01*image_width) or (image_width - box_width < 0.01*image_width)) or ((box_y < 0.01*image_height) or (image_height - box_height < 0.01*image_height))):
                    print("Move Away from the book")
                    i=0
                else :
                    print("Perfect")
                    i=i+1
            if(i==15 and not taken):
                taken = True
                cv2.imwrite('book.png',image[int(box_y):int(box_height),int(box_x):int(box_width)])
                break
            cv2.putText(image,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('image', resized)
    # cv2.imwrite("image_box_text.jpg",image)
    cv2.waitKey(1)
cv2.destroyAllWindows()

pic= image[int(box_y):int(box_height),int(box_x):int(box_width)]

####

#im_1,im_2,im_3=cv2.split(pic)
#im__1=im_1.transpose()
#im__3=im_3.transpose()
#im__2=im_2.transpose()
#pic=cv2.merge((im__1,im__2,im__3))

#####
###
"""
rows,cols = pic.shape[0:2]

M = cv2.getRotationMatrix2D((cols,rows),270,1)
pic= cv2.warpAffine(pic,M,(cols,rows))

"""


croppic1= per.pertransform1(pic)
croppic2= per.pertransform2(pic)
text1= pyt.image_to_string(croppic1)
text2= pyt.image_to_string(croppic2)
al= list(range(48,58))
bl= list(range(65,91))
cl= list(range(97,123))
dl=al+bl+cl
il=0
jl=0
for ip in text1:
	if ord(ip) in dl:
		il=il+1
for jp in text2:
	if ord(jp) in dl:
		jl=jl+1

if jl>il:
	text=text2
else:
	text=text1
f= open("text.txt","w+")
f.write(text)
f.close()
print(text)

string=alp.translate(text)
biv.binarycontent(string)
