import numpy as np
import copy

#Variance matrices experimentally determined
MODEL_ERR_X_STD = 1.753999636569676
MODEL_ERR_Y_STD = 2.5832717319349294
MODEL_ERR_ALPHA_STD =  0.03738207502328924
MODEL_ERR_V_STD = 0.027805629538742476


SENSOR_ERR_X_STD = 0.20344381248915722
SENSOR_ERR_Y_STD = 0.7420408115479226 
SENSOR_ERR_ALPHA_STD =  0.010427969435971924
SENSOR_ERR_V_STD = 3.127273511234235
R=np.diag([SENSOR_ERR_X_STD, SENSOR_ERR_Y_STD, SENSOR_ERR_ALPHA_STD, SENSOR_ERR_V_STD])
Q=np.diag([MODEL_ERR_X_STD, MODEL_ERR_Y_STD, MODEL_ERR_ALPHA_STD, MODEL_ERR_V_STD])



def extended_kalman_filter(x_est, u, P_est, measurement_cam, measurement_v, T, h):
    """
    x_est = [x, y, alpha, v]  , previous estimated state
    u = [target_l, target_r]  , input applied on wheels
    measurement_cam = [x, y, alpha]  , camera measurement
    measurement v = [speed_l, speed_r]  , speed sensed on wheels 
    """
    
    # converts pixels to mm based on camera height
    ratio_pixel2mm=h/69 
    measurement_cam = [measurement_cam[0][0]*ratio_pixel2mm, measurement_cam[0][1]*ratio_pixel2mm, measurement_cam[1]]
        
    # converts thymio speed to mm/s 
    convert_speed_to_mms=0.43478260869565216
    measurement_v = [measurement_v[0] * convert_speed_to_mms, measurement_v[1] * convert_speed_to_mms]

    # converts control input speed to mm/s 
    u= [u[0] * convert_speed_to_mms, u[1] * convert_speed_to_mms]

    d=143 # mm, distance between wheels

    # Constant R is recomputed if the robot is not detected, so we made a copy of it
    RR = R.copy()
    
    if measurement_cam[0]<0 and measurement_cam[1]<0: # (x,y) = (-1,-1) : camera did not find Thymio
        # We can only sense the speed on wheels -> observation matrix C observes only speed v
        #                                       -> measurement are only the speed sensed
        #                                       -> the variance on measuremnts in only the velocity one
        C = np.array([[0, 0, 0, 1]])
        y = np.array([np.mean(measurement_v)])
        RR = np.array([[R[3][3]]])
    else:
        # Full knowledge on states thanks to sensors on each state
        C = np.identity(4)
        y = np.array([[measurement_cam[0]], [measurement_cam[1]], [measurement_cam[2]], [np.mean(measurement_v)]])

    # A priori prediction : x(k+1) = f(x(k), u(k))
    x_a_priori = np.array([[float(x_est[0]+T*x_est[3]*np.cos(x_est[2]))], [float(x_est[1]+T*x_est[3]*np.sin(x_est[2]))], [float(x_est[2]+T*(u[0]-u[1])/d)], [float((u[0]+u[1])/2)]])
    A = np.array([[1, 0, float(-T*x_est[3]*np.sin(x_est[2])), float(T*np.cos(x_est[2]))], [0, 1, float(T*x_est[3]*np.cos(x_est[2])), float(T*np.sin(x_est[2]))], [0, 0, 1, 0], [0, 0, 0, 0]]) # linearized version of the dynamic

    # Update state uncertainties
    P_a_priori=np.dot(np.dot(A,P_est), A.T) + Q
    
    # Introduce correction based on measurements
    i = y - np.dot(C, x_a_priori)
    S = np.dot(np.dot(C, P_a_priori), C.T) + RR
    K = np.dot(P_a_priori, np.dot(C.T, np.linalg.inv(S)))

    # Update a priori states, uncertainties with measurements
    next_x_est = x_a_priori + np.dot(K, i)
    next_P_est = np.dot(np.identity(4) - np.dot(K, C), P_a_priori)

    # converts mm to pixel based on camera height
    pose_est = ((int(x_est[0]/ratio_pixel2mm), int(x_est[1]/ratio_pixel2mm)), float(x_est[2]))
    
    return next_x_est, next_P_est, pose_est



