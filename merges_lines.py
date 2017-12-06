##import cv2
##import numpy as np
##
##gray = cv2.imread('scrnshot.png')
##mser = cv2.MSER_create()
##
###Resize the image so that MSER can work better
##gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
##
###gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##vis = gray.copy()
##
##regions = mser.detectRegions(gray)
##hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
##cv2.polylines(vis, hulls, 1, (0,255,0)) 
##
##cv2.namedWindow('img', 0)
##cv2.imshow('img', vis)
##while(cv2.waitKey()!=ord('q')):
##    continue
##cv2.destroyAllWindows()
import cv2
import numpy as np
from PIL import Image

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

img = cv2.imread('sample_screenshots/testscrnshot.png')
img_cutversion=Image.open('sample_screenshots/testscrnshot.png')

mser = cv2.MSER_create()

#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy() #this is what we see

regions = mser.detectRegions(gray)

#hulls is a list of np.arrays that are the points
#each array has tuples in them specifying the coordinates
#hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
rects = []
hulls = []
for p in regions[0]:
    x,y,w,h= cv2.boundingRect(p.reshape(-1, 1, 2)) 
 #   rect=cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2) #draws the rectangles on vis
    rects.append([x,y,w,h])

rects.sort(key=lambda x: x[1])
summarized_rects = [rects[0]]

for r in range(1,len(rects)):
    this = rects[r]
    prev = summarized_rects[-1]

    #print (this)

    state, average = overlaps(prev, this)
    #print(state, average)
    #print( prev, this)
    if state == "overlap":
        #print("---contains", average)
        summarized_rects[-1]=average
    if state == "disjoint":
        #print("--disjoint", prev, this)
        summarized_rects.append(average)

print ("summarized rects", summarized_rects)
print("pillow image size", img_cutversion.size)

cutimages=[]
for r in summarized_rects:
    x,y,w,h = r
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2) #draws it

    cutimages.append(img_cutversion.crop((x//2,y//2,(x+w)//2,(y+h)//2)))

for index in range(min(100,len(summarized_rects))): #set the min to 100 for now to avoid oversaving
    picture = cutimages[index]
    picture.save('image' + str(index) +'.png')

 #   print([x,y,w,h])
 #   box = cv2.boxPoints(rect)
#    box = np.int0(box)
 #   cv2.drawContours(vis,[box],0,(0,0,255),2)
#print("hulls",hulls)
#cv2.polylines(vis, hulls, 1, (0,255,0)) 


#display image with boxes around letters
 
cv2.namedWindow('img', 0)
cv2.imshow('img', vis)
##
##mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
##for contour in hulls:
##    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#display the image with mask, show only the letters
##text_only = cv2.bitwise_and(img, img, mask=mask)
##cv2.namedWindow('text', 0)
##cv2.imshow('text', text_only)

#press q to close window
while(cv2.waitKey()!=ord('q')):
    continue
cv2.destroyAllWindows()



###
#code for getting only letters
#https://stackoverflow.com/questions/44185854/extract-text-from-image-using-mser-in-opencv-python

###############
#puts green boxes around things

##import cv2
##
##img = cv2.imread('scrnshot.png')
##mser = cv2.MSER_create()
##
###Resize the image so that MSER can work better
##img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
##
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##vis = img.copy()
##
##regions = mser.detectRegions(gray)
##hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
##cv2.polylines(vis, hulls, 1, (0,255,0)) 
##
##cv2.namedWindow('img', 0)
##cv2.imshow('img', vis)
##while(cv2.waitKey()!=ord('q')):
##    continue
##cv2.destroyAllWindows()
###############
#cuts part of an image
##
##from PIL import Image
##i = Image.open('scrnshot.png')
##w,h = i.size
##row_to_sum = []
##for row in range(h):
##    row_sum = 0
##    for col in range(w):
##        row_sum += sum(i.getpixel((col,row))) #gets the value at that pixel
##    row_to_sum.append(row_sum) #appends the sum of that row
##
###high value: white; low value: black
##
###now get the changes.
##print(row_to_sum)
##frame2 = i.crop(((275, 0, 528, 250)))
##frame2.save('cut.png')
