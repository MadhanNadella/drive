import cv2
import perspective as per
import pytesseract as pyt
import binvalues as biv
import alphaToBraille as alp

pic=cv2.imread("/home/madhan/EE04_Tess/pic54.png")
croppic1= per.pertransform1(pic)
croppic2= per.pertransform2(pic)
text1= pyt.image_to_string(croppic1)
#print(text1)

text2= pyt.image_to_string(croppic2)
#print(text2)
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
print(jl)
print(il)
#print(text)
string=alp.translate(text)
biv.binarycontent(string)
