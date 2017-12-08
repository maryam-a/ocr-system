import numpy as np
from numpy.random import randint, choice
import os
import string
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
np.random.seed(123)

# Black against white (huge initial) or white against black
# Monospace fonts: Consolas, Courier New, Lucida Regular Console, UbuntuMono-R, Inconsolata
TF = ImageFont.truetype('consola.ttf', 18)

def MakeImg(t, f, fn, s = (100, 100), o = (0, 0)):
    '''
    Generate an image of text
    t:      The text to display in the image
    f:      The font to use
    fn:     The file name
    s:      The image size
    o:      The offest of the text in the image
    '''
    img = Image.new('RGB', s, "white")
    draw = ImageDraw.Draw(img)
    draw.text((0,1), t, (0,0,0), font = f)
    img.save(fn)
    
def GetFontSize(S):
    img = Image.new('RGB', (1, 1), "white")
    draw = ImageDraw.Draw(img)  
    return draw.textsize(S, font = TF)

def GenSingleLine(MINC = 10, MAXC = 64, NIMG = 128, DP = 'Out'):  
    createDirIfNotPresent(DP)
    #The possible characters to use
    CS = list(string.ascii_letters) + list(string.digits)
    MS = GetFontSize('\n'.join(['0' * MAXC]))   #Size needed to fit MAXC characters
    print(MS)
    
    Y = []
    for i in range(NIMG):               #Write images to ./Out/ directory
        Si = ''.join(choice(CS, randint(MINC, MAXC)))
        FNi = str(i) + '.png'
        MakeImg(Si, TF, os.path.join(DP, FNi), MS)
        Y.append(FNi + ',' + Si)
    with open('Train2.csv', 'w') as F:   #Write CSV file
        F.write('\n'.join(Y))
        
def GenMultiLine(ML = 5, MINC = 10, MAXC = 64, NIMG = 128, DP = 'Img'):
    createDirIfNotPresent(DP)
    #The possible characters to use
    CS = list(string.ascii_letters) + list(string.digits)
    MS = GetFontSize('\n'.join(ML * ['0' * MAXC]))
    Y = []
    for i in range(NIMG):               #Write images to ./Img/ directory
        temp = list(''.join(choice(CS, randint(MINC, MAXC))) for _ in range(randint(1, ML + 1)))
        Si = '\n\n'.join(temp)
        for_csv = '\\n'.join(temp)
        FNi = str(i) + '.png'
        MakeImg(Si, TF, os.path.join(DP, FNi), MS)
        Y.append(FNi + ',' + for_csv)
    with open('Train.csv', 'w') as F:   #Write CSV file
        F.write('\n'.join(Y))

def createDirIfNotPresent(dirName):
    # ensure save location is present
    if not Path(dirName).is_dir():
        Path(dirName).mkdir()


        
if __name__ == "__main__":
    GenMultiLine()
    GenSingleLine(NIMG = 1024)