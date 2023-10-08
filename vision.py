
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt  

## -----------------Some constants-------------------##

#default constant use to detect red, green and blue in HSV
LOWER_RED = [168, 50, 50] 
UPPER_RED = [178, 255, 255]

LOWER_GREEN = [66, 50, 50]
UPPER_GREEN = [76, 255, 255]

LOWER_BLUE = [108, 50, 50]
UPPER_BLUE = [118, 255, 255]
THRESHOLD_COLOR = [LOWER_RED, UPPER_RED, LOWER_GREEN, UPPER_GREEN, LOWER_BLUE, UPPER_BLUE]

# Size in pixel of landmark drawn in vide0
ROBOT_MARK_RADIUS = 15

# Threshold to convert grayscale image in white and black image
THRESHOLD = 40 

# OFFSET valut to dilate obstacle in global navigation
OFFSET = 150

# Obstacle contour sample period 
COEF_SAMPL = 12

## -----------------function use to control the installation of the camera and the thresholds values-------------------##

#INPUT (Image in BRG, threshold value for each color in HSV)
#OUTPUT (image with white where red detected, image with white where green detected, image with white where blue detected)
def landmarkDetectionTest(frame, threshold):
    _,fr = getPosColor(frame, 'r', threshold)
    _,fg = getPosColor(frame, 'g', threshold)
    _,fb = getPosColor(frame, 'b', threshold)

    return fr,fg,fb


## -----------------functions use the vision-------------------##

## This function extract all zone with color 
#INPUT (Image in BRG, color to detect: 'r', 'g' or 'b', threshold value for each color in HSV)
#OUTPUT (White and black image with white where specified color is detected)
def getColorShape(frame, color, threshold): 
       
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if color == 'b':
        lower = np.array(threshold[4])
        upper = np.array(threshold[5])
    elif color == 'r':
        lower = np.array(threshold[0])
        upper = np.array(threshold[1])
    elif color == 'g':
        lower = np.array(threshold[2])
        upper = np.array(threshold[3])

    frame = cv2.inRange(frame, lower, upper)
    frame = cv2.medianBlur(frame, 21)
    
    return frame


## This function give the position of the center of zone of color 'color'
#INPUT (Image in BRG, color to detect: 'r', 'g' or 'b', threshold value for each color in HSV)
#OUTPUT (Position where specified landmark is detected, White and black image with white where specified color is detected)
def getPosColor(frame, color, threshold = THRESHOLD_COLOR):

    img = getColorShape(frame, color, threshold)

    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if not (len(contours) == 1):
        return (-1, -1), img #if more than one landmark is detected, we can't tell which one correspond to the robot or to the goal
    else:
        M = cv2.moments(contours[0])
        try:
             px = int(M["m10"] / M["m00"])
        except ZeroDivisionError:
            return (-1, -1), img # if the detected landmark is 1 dimensional it's not the robot
        
        try:
            py = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            return (-1, -1), img
        
        return (px, py), img


## This function give the position of the robot
#INPUT (Image in BRG, threshold use to detect color)
#OUTPUT (Position and orientation of the robot in format (posX, posY), alpha). Return ((-1,-1),-1) if the robot is not detected
def getPose(frame, threshold = THRESHOLD_COLOR):
        
    posR, _ = getPosColor(frame, 'r', threshold)
    posG, _ = getPosColor(frame, 'g', threshold)
    
    if (posG[0] == -1) or (posR[0] == -1):
        return ((-1,-1), -1)

    alpha = np.arctan2(posG[1]-posR[1],posG[0]-posR[0])
    
    return (posR, alpha)

## This function give the position of the robot
#INPUT (Image in BRG, threshold use to detect color)
#OUTPUT (Position and orientation of the robot in format (posX, posY), alpha). Return ((-1,-1),-1) if the robot is not detected
def getGoal(frame, threshold = THRESHOLD_COLOR):
        
    pos, _ = getPosColor(frame, 'b', threshold)

    if (pos[0] == -1):
        print("goal not detected by camera")
        return (-1,-1)
        
    return pos

## This function return a white and black image with obstacle in black
#INPUT (Image in BRG or gray scale, position of the goal)
#OUTPUT white and black image with obstacle in black
def getObstacleForm(img, pGoal):
    if len(img.shape) == 3: # determine if the image is in gray scale or not
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    (wi, hi) = img.shape # mask the goal to not detect it like an obstacle
    mask = np.zeros((wi, hi), dtype="uint8")
    cv2.circle(mask, pGoal, int(OFFSET/3.5), 255, -1)
    cv2.bitwise_not(img, img, mask=mask)

    _, img = cv2.threshold(img,THRESHOLD,255,cv2.THRESH_BINARY)
    img = cv2.medianBlur(img, 15) # remove the noise of the background
    return img    


## This function samples the contour of each obstacles
#INPUT (Image in BRG or gray scale, position of the goal)
#OUTPUT (list of point, white and black image with obstacle in black, white and black image with obstacle in black and offset)
def getCrossingPoint(img, posGoal):
    w,h,_ = img.shape
    w -= 1
    h-=1
    
    imageObstacle = getObstacleForm(img, posGoal)
    
    #dilate the obstacle 
    mask = np.zeros((OFFSET, OFFSET), np.uint8)
    cv2.circle(mask, (int(OFFSET/2), int(OFFSET/2)), int(OFFSET/2), 1, -1)
    dilObs = cv2.erode(imageObstacle,mask,iterations = 1)

    #find the contour of obstacles
    contours = cv2.findContours(dilObs, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    

    #samples the contour
    point_contour_echantillonne = []
    for contour in contours[0][:]:
        for i in range(math.floor((len(contour)-1)/COEF_SAMPL)):
            #Do not detect camera border as an contour
            if (contour[COEF_SAMPL*i][0][0] == 0 or contour[COEF_SAMPL*i][0][0] == h or contour[COEF_SAMPL*i][0][1] == 0 or contour[COEF_SAMPL*i][0][1] == w):
                continue
            point_contour_echantillonne.append((contour[COEF_SAMPL*i][0][0], contour[COEF_SAMPL*i][0][1]))
        
    return point_contour_echantillonne, imageObstacle, dilObs

## This function determine if it's possible to reach one point from an other in straight line without crossing an obstacle
#INPUT (Point1, Point2, Frame of obstacles with the offset)
#OUTPUT (True or False)
def isCrossingObs(p1, p2, dilObs):
    line = (np.linspace(p1, p2, 30)).astype(int)
    for p in line:
        if dilObs[p[1], p[0]] == 0:
            return True
    return False

## This function determine all the possible paths
#INPUT (List of point from contours, frame of the obstacle )
#OUTPUT [(P1, [all the points reachable from P1]), (P2, [all the points reachable from P2]), ...]
def getPossiblePath(posCorner, imageObstacle):
    
    offset = OFFSET-10
    mask = np.zeros((offset, offset), np.uint8)
    cv2.circle(mask, (int(offset/2), int(offset/2)), int(offset/2), 1, -1)
    
    dilObs = cv2.erode(imageObstacle,mask,iterations = 1)
    possiblePath = []
    
    for p1 in posCorner:
        canReach=[]
        for p2 in posCorner:
            if (p1 == p2) or isCrossingObs(p1,p2, dilObs):
                continue
            canReach.append(p2)
        possiblePath.append((p1, canReach))
    return possiblePath

## This function determine all the reachable points from the robot position
#INPUT (List of point from contours, position of robot, frame of the obstacle )
#OUTPUT [(P1, [all the points reachable from P1]), (P2, [all the points reachable from P2]), ...]
def getPossiblePathByRobot(posCorner, poseRobot, imageObstacle):
    
    offset = OFFSET-10
    mask = np.zeros((offset, offset), np.uint8)
    cv2.circle(mask, (int(offset/2), int(offset/2)), int(offset/2), 1, -1)
    
    dilObs = cv2.erode(imageObstacle,mask,iterations = 1)
    

    canReach=[]
    for p in posCorner:
        if isCrossingObs(poseRobot,p, dilObs):
            continue
        canReach.append(p)
    possiblePathByRobot = [(poseRobot, canReach)]
    return possiblePathByRobot


## -----------------function use to add some information on the image get from the camera-------------------##

## This function add on image all the possible path
#INPUT (List of all possible path, image get by the camera)
#OUTPUT [image with all possible path]
def drawPossiblePath(possiblePath, image):
    for three in possiblePath:
        for p in three[1]:
            cv2.line(image,three[0],p,100,1)
    return image  

## This function add on image all the path path for the robot
#INPUT (list of points, image get by the camera)
#OUTPUT [image with the path]
def drawBestPath(bestPath, image):
    for i in range(len(bestPath)-1):
        cv2.line(image,bestPath[i],bestPath[i+1],[0, 100,0 ],4)
    return image

## This function add on the image a symbol for the robot and the goal
#INPUT (image get by the camera, robot pose, goal position, Boolean (True = draw goal, False= do not draw goal))
#OUTPUT [image with the new symbol]
def drawRobotAndGoal(image, pose, posGoal = (-2,-2), drawGoal = False):
    cv2.line(image, pose[0], (pose[0][0] + int(30*math.cos(pose[1])), pose[0][1] + int(50*math.sin(pose[1]))) , (223, 3, 0), 3) 
    cv2.circle(image, pose[0], ROBOT_MARK_RADIUS, (255, 0, 0), 2)
    if drawGoal and posGoal[1] != -2:
        cv2.circle(image, posGoal, ROBOT_MARK_RADIUS, (255, 255, 0), 2)
    return image



## This function display the zone detected for each color, its only purpose is debugging 
#INPUT (image get by the camera, threshold value for each color)
#OUTPUT display 4 images
def testColor(frame , thresholdColor):
    _,fr = getPosColor(frame, 'r', thresholdColor)
    _,fg = getPosColor(frame, 'g', thresholdColor)
    _,fb = getPosColor(frame, 'b', thresholdColor)
    plt.imshow(frame)

    fig = plt.figure(figsize = (20, 10))
    plt.subplot(131),plt.imshow(fr, cmap = 'gray')
    plt.title('Read zone'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(fg,cmap = 'gray')
    plt.title('green zone'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(fb,cmap = 'gray')
    plt.title('blue zone'), plt.xticks([]), plt.yticks([])

