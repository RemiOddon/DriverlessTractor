import numpy as np
import math


## this function compute 7 point in front of the robot.
#INPUT (Robot position)
#OUTPUT (list of the  points)
def getPoints (pose) :
    
    a = (1/6)*math.pi # Angle between each points
    R = 80 # Radius form the Robot to the point
    Dl = 25 #Offset to get the center of the robot 

    points = []
    for i in [-3,-2,-1, 0, 1, 2, 3]:
        x = int(pose[0][0] + R*math.cos(-i*a + pose[1]) + Dl*math.cos(pose[1]))
        y = int(pose[0][1] + R*math.sin(-i*a + pose[1]) + Dl*math.sin(pose[1]))
        points.append([x,y])
            
    return points


## this function determine if it exist some global obstacle in front of the robot.
#INPUT (Robot position, white and black image of the obstacle)
#OUTPUT (list of one or zero)
def determineWeights (pose, imageObstacleDil) :

    points = getPoints(pose)
    weights = np.zeros (7)
    
    for i in range (7) :
        
        x = points [i][0]
        y = points [i][1]
        try:
            pixelValue = imageObstacleDil[y][x] 
        except IndexError:
            pixelValue = 0 # the bounds of map are unreachable

            
        if pixelValue == 0 : #Noir donc il y a un obstacle 
            weights[i] = 1
        
        if pixelValue == 255 : #Blanc donc pas d'obstacle
            weights[i] = 0
            
    return weights


## this function determine in which state the robot have to be depending on obstacle around itself
#INPUT (Robot position, current state, value measure by the robot proximity sensor,  white and black image of the obstacle with the offset)
#OUTPUT (new sate (1 = global nav, 2 = local nav))
def test_state(pose, state, prox_horizontal, imageObstacleDil):

    #Threshold use to determine if we detect or not an obstacle. 
    Tresh_obst_in = 1
    
    if state == 1:
        #if any obstacle are detected with prox sensor, the new state is 2
        for i in range(5):
            if (prox_horizontal[i] > Tresh_obst_in):                
                state = 2
        return state
    if state == 2:
        # if the front prox sensor detect nothing, and the robot are not to close of an global obstacle, the new state = 1
        if sum(prox_horizontal[0:5]) == 0 and imageObstacleDil[pose[0][1]][pose[0][0]] == 255:
                state = 1
        return state
    return state


## this function determine the best control to avoid local and global obstacles during local navigation
#INPUT (Robot position, value measure by the robot proximity sensor,  white and black image of the obstacle)
#OUTPUT (speed of left motor, speed of right motor)
def prox(pose, prox_horizontal, imageObstacleDil):

    #Weight for the neural network
    w_l = [80,  100, -15, -100, -80, 0,0, -5,-15,  -7, -5, 7, 15, 5]
    w_r = [-80, -100, -14,  100,  80, 0,0, 5, 15, 7, -5,  -7,  -15, -5]
    
    # virtual sensing of the global obstacle
    global_sens = determineWeights (pose, imageObstacleDil)
    
    # Scale factors for sensors
    sensor_scale = 1000
    global_scale = 9 # à tuner

    # Get and scale inputs from sensors
    x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    for i in range(5):
        #if prox_horizontal[i] > 2000:
        x[i] = prox_horizontal[i] / sensor_scale
    for i in range(7):
        x[i+7] = global_sens[i] * global_scale # i + 7

    y = [100,100]    # mean speed

    for i in range(len(x)):    
        # Compute outputs of neurons and set motor powers
        y[0] = y[0] + x[i] * w_l[i]
        y[1] = y[1] + x[i] * w_r[i]

    
    # If the speed is too big, we normalize it.
    ym = np.max(np.abs(y))
    if ym > 200:
        y[0] = y[0]/ym*200
        y[1] = y[1]/ym*200
       
    return y[0], y[1]