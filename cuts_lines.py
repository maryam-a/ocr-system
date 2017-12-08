
import cv2
import numpy as np
from PIL import Image
np.random.seed(123)

#checks whether two images overlap. If they do, return a larger bounding box.
def overlaps(prev, curr):
    
    x1,y1,w1,h1=prev
    x2,y2,w2,h2=curr

    x1_inside_x2 = x2<=x1 and x1<=x2+w2
    x2_inside_x1 = x1<=x2 and x2<=x1+w1
    y1_inside_y2 = y2<=y1 and y1<=y2+h2
    y2_inside_y1 = y1<=y2 and y2<=y1+h1

    #if they overlap
    if  (y1_inside_y2 or y2_inside_y1):
        return ("overlap",
                 [min(x1,x2), min(y1,y2), max(x1+w1,x2+w2)-min(x1,x2)
                  , max(h1+y1,h2+y2)-min(y1,y2)])
    else:
        return ("disjoint", curr)

#img is image for opencv; img_cutversion is for Pillow to cut
img = cv2.imread('Img/0.png')
img_cutversion=Image.open('Img/0.png')

mser = cv2.MSER_create()

#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy() #this is what we see

regions = mser.detectRegions(gray)

#gets all the regions detected by mser
rects = []
for p in regions[0]:
    x,y,w,h= cv2.boundingRect(p.reshape(-1, 1, 2)) 
    rects.append([x,y,w,h])

#sorts the rectangles based on the y position (row of the picture)
rects.sort(key=lambda x: x[1])

#now combines letters along each row
summarized_rects = [rects[0]]
for r in range(1,len(rects)):
    this = rects[r]
    prev = summarized_rects[-1]
    state, average = overlaps(prev, this)
    if state == "overlap":
        summarized_rects[-1]=average
    if state == "disjoint":
        summarized_rects.append(average)

#gets an array of the cut images using pillow
cutimages=[]
for r in summarized_rects:
    x,y,w,h = r
    cv2.rectangle(vis,(x,y),(x+1280,y+28),(0,255,0),2) #draws it to visualize

    #cuts the image. remember we scaled up the picture for MSER, so divide by 2
    #for pillow to cut.
    cutimages.append(img_cutversion.crop((x//2,y//2,(x+1280)//2,(y+28)//2)))

#save each of the images in cutimages
for index in range(min(100,len(summarized_rects))): #set the min to 100 for now to avoid oversaving
    picture = cutimages[index]
    picture.save('test/image' + str(index) +'.png')


#display the image with green boxes around what we've cut
cv2.namedWindow('img', 0)
cv2.imshow('img', vis)


#press q to close window
while(cv2.waitKey()!=ord('q')):
    continue
cv2.destroyAllWindows()



###
#code for getting only letters
#https://stackoverflow.com/questions/44185854/extract-text-from-image-using-mser-in-opencv-python
