import numpy as np
import math
from scipy.spatial import distance


MIN_DIST = 50 # minimum distance to consider we are in a given point

## this function put an angle in the range [-pi, pi]
#INPUT (angle in radian)
#OUTPUT (angle between [-pi, pi])
def toMPiPi(angle):
    angle = angle % (2*np.pi)
    if angle > np.pi:
        angle = angle - 2*np.pi
    return angle


## this function determine as set of velocity for each motor in sort to reach a point
#INPUT (robot position, point to reach, boolean value (True if the angular error was too big the past iteration, false otherwise))
#OUTPUT (angle between [-pi, pi])
def reachPoint(pose, currentPoint, memAlpha):
    
    # Compute the angular error
    pR = np.asarray(pose[0])
    cP = np.asarray(currentPoint)
    angularError = toMPiPi(pose[1]-np.arctan2(cP[1]-pR[1],cP[0]-pR[0]))

    #mean velocity
    vm = 100

    #Determine if the error is too big, if it's case the robot must only correct is angular error 
    if abs(angularError) > math.pi/1.5: 
        memAlpha = True
        vm = 0
    if memAlpha: # memory: if the error was to big the previous iteration, the robot continue to turn on it self until angular error<pi/4
        if abs(angularError) > math.pi/4:
            vm = 0
        else:
            memAlpha = False
            vm = 100
    dv = 200*angularError

    dv = np.sign(dv)*min(abs(dv),100) #bounded the differential speed
    vl = vm-dv
    vr = vm+dv
    
    return vl, vr, memAlpha



## This function determine if the robot reach a given point
#INPUT (point, robot position)
#OUTPUT (boolean value)
def isPointReach(currentPoint, pos):
    if abs(distance.euclidean(currentPoint, pos)) < MIN_DIST:
        return True
    return False
    