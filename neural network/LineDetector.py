import cv2
import numpy as np


#this function is not used 

def findLine(image,num):
#set up the size of the occupancy map
    
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    img = cv2.cvtColor(open_cv_image, cv2.cv.CV_BGR2RGB)
        
    # img = cv2.imread('temp.png')
        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


    lines = cv2.HoughLines(closed,1,np.pi/180,50)


    if num == 1:
        if lines != None:
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,0,250),2)
    
        cv2.imshow('1.jpg',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if lines != None:
        return 'line found'
    else:
        return 'no line found'
        # print 'no lines'
            # return False
    





