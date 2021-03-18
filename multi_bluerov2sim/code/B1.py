from pymavlink import mavutil
from mpl_toolkits.mplot3d import Axes3D
from pynverse import inversefunc
from multiprocessing import Process

import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

## Quaternion to Euler ##
def q2e(q):

	    # roll
        roll = math.atan2( 2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2) )

        # pitch
        sinp = -2.0 * (q[3] * q[1] - q[2] * q[0])
        if math.fabs(sinp) >= 1:
            # use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

		# yaw
        yaw = math.atan2( 2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2) )

        rpy = np.array([ [roll], [pitch], [yaw] ])

        return rpy

## Euler to Quaternion ##
def e2q(rpy):

            cr = math.cos(rpy[0] / 2)
            sr = math.sin(rpy[0] / 2)
            cp = math.cos(rpy[1] / 2)
            sp = math.sin(rpy[1] / 2)
            cy = math.cos(rpy[2] / 2)
            sy = math.sin(rpy[2] / 2)

            q = [0, 0, 0, 0]

            q[0] = cr * cp * cy + sr * sp * sy # w
            q[1] = sr * cp * cy - cr * sp * sy # x
            q[2] = cr * sp * cy + sr * cp * sy # y
            q[3] = cr * cp * sy - sr * sp * cy # z

            q = np.array([ [q[0]], [q[1]], [q[2]], [q[3]] ])

            return q

def pwm_to_thrust(pwm):
        """Transform pwm to thruster value
        The equation come from:
            https://colab.research.google.com/notebook#fileId=1CEDW9ONTJ8Aik-HVsqck8Y_EcHYLg0zK

        Args:
            pwm (int): pwm value

        Returns:
            float: Thrust value
        """
        return -3.04338931856672e-13*pwm**5 \
            + 2.27813523978448e-9*pwm**4 \
            - 6.73710647138884e-6*pwm**3 \
            + 0.00983670053385902*pwm**2 \
            - 7.08023833982539*pwm \
            + 2003.55692021905

def go_point(master,DesX,DesY,DesZ,DesYaw):

	## bluerov2 data
	m = 8.44    # kg
	W = 82.80   # N
	B = 84.76   # N
	Ix = 0.16   # kg*m^2
	Iy = 0.22   # kg*m^2
	Iz = 0.13   # kg*m^2
	zg = -0.037 # m

	Xud = -5.5   # kg
	Yvd = -12.7  # kg
	Zwd = -14.57 # kg
	Kpd = -0.12  # kg*m^2/rad
	Mqd = -0.12  # kg*m^2/rad
	Nrd = -0.12  # kg*m^2/rad

	Xu = -4.03   # N*s/m
	Yv = -6.22   # N*s/m
	Zw = -5.18   # N*s/m
	Kp = -0.07   # N*s/rad
	Mq = -0.07   # N*s/rad
	Nr = -0.07   # N*s/rad

	Xuu = -18.18 # N*s^2/m^2
	Yvv = -21.66 # N*s^2/m^2
	Zww = -36.99 # N*s^2/m^2
	Kpp = -1.55  # N*s^2/rad^2
	Mqq = -1.55  # N*s^2/rad^2
	Nrr = -1.55  # N*s^2/rad^2

	## PID
	KP = np.array([   [3],   [3],   [3],   [4],   [4],   [2] ])
	KI = np.array([ [0.2], [0.2], [0.2], [0.3], [0.3], [0.1] ])
	KD = np.array([ [2.5], [2.5], [0.5], [0.5],   [1], [0.5] ])
	IControl = 0

	## Sampling time(RK)
	h = 0.14*(0.052)
	t = 0

	## Inirial Previous error
	errorPrevious = np.zeros((12,1))
	
	## Destination Point & Inital Point
	DP = np.array([[DesX], [DesY], [DesZ], [0], [0], [DesYaw], [0], [0], [0], [0], [0], [0]]) # destination point
	I  = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])               # initial point

	while(1):
	#for i in range (0, 1) :
	    start_time = time.time()
	    error = (DP-I)
        
        ## position error
	    error_p = error[0:6]         

        ## position error(quaternion)
	    error_pq = np.array([ error[0], 
	    				      error[1], 
	    				      error[2], 
	    				      e2q(error[3:6])[0], 
	    				      e2q(error[3:6])[1], 
	    				      e2q(error[3:6])[2], 
	    				      e2q(error[3:6])[3] ])                      

	    ## previous position error 
	    errorPrevious_p  = np.array( errorPrevious[0:6] )     

        ## previous position error(quaternion)
	    errorPrevious_pq = np.array([ errorPrevious_p[0], 
	    							  errorPrevious_p[1], 
	    							  errorPrevious_p[2], 
	    							  e2q(errorPrevious_p[3:6])[0], 
	    							  e2q(errorPrevious_p[3:6])[1], 
	    							  e2q(errorPrevious_p[3:6])[2], 
	    							  e2q(errorPrevious_p[3:6])[3]  ])   
	    
	    ## Current Value [World frame]
	    x = float(I[0])  
	    y = float(I[1]) 
	    z = float(I[2]) 
	    roll = float(I[3])
	    pitch = float(I[4]) 
	    yaw = float(I[5]) 
	    u = float(I[6])
	    v = float(I[7]) 
	    w = float(I[8]) 
	    p = float(I[9]) 
	    q = float(I[10]) 
	    r = float(I[11])  

	    E2Q= e2q(I[3:6])
	    qj = float(E2Q[0])
	    E1 = float(E2Q[1])
	    E2 = float(E2Q[2])
	    E3 = float(E2Q[3])

	    R = np.array([  [  1-2*(E2**2 + E3**2),        2*(E1*E2 - E3*qj),         2*(E1*E3 + E2*qj)],
	                    [    2*(E1*E2 + E3*qj),    1 - 2*(E1**2 + E3**2),         2*(E2*E3 - E1*qj)],
	                    [    2*(E1*E3 - E2*qj),        2*(E2*E3 + E1*qj),     1 - 2*(E1**2 + E2**2)]  ])

	    T = (0.5) * np.array([ [-E1, -E2, -E3],
	                           [ qj, -E3,  E2],
	                           [ E3,  qj, -E1],
	                           [-E2,  E1,  qj]   ])  
	    
	    J = np.hstack([ np.vstack([R, np.zeros((4,3))]),  np.vstack([np.zeros((3,3)), T]) ])

	    ## error rotation [World -> Body]
	    error_p_b = np.dot(np.linalg.pinv(J), error_pq)

	    errorPrevious_p_b = np.dot(np.linalg.pinv(J), errorPrevious_pq)

	    PControl = KP * error_p_b  
	    #IControl = IControl + KI * error_p_b*h  
	    DControl = KD * (error_p_b - errorPrevious_p_b)/h 
	    PID = PControl #+ IControl + DControl

	    PID_4 = np.vstack((PID[0:3], PID[5]))
	    PID_X = PID_4[0] 
	    PID_Y = PID_4[1]
	    PID_Z = PID_4[2]
	    PID_Yaw = PID_4[3]
	    '''
	    if abs(PID_X) > 50 :
	    	PID_X = 50 * (PID_X) / abs(PID_X)

	    if abs(PID_Y) > 50 :
	    	PID_Y = 50 * (PID_Y) / abs(PID_Y)

	    if abs(PID_Z) > 50 :
	    	PID_Z = 50 * (PID_Z) / abs(PID_Z)

	    if abs(PID_Yaw) > 50 :
	    	PID_Yaw = 50 * (PID_Yaw) / abs(PID_Yaw)
	    '''
	    ## Thruster to pwm
	    pwm_2_thr   = (lambda x: -3.04338931856672e-13*x**5 + 2.27813523978448e-9*x**4 - 6.73710647138884e-6*x**3 + 0.00983670053385902*x**2 - 7.08023833982539*x + 2003.55692021905)
	    PID_pwm_X   = (inversefunc(pwm_2_thr,y_values = PID_X/9.8) - 1500)  /(0.4) + 12.064 
	    PID_pwm_Y   = (inversefunc(pwm_2_thr,y_values = PID_Y/9.8) - 1500)  /(0.4) + 12.064  
	    PID_pwm_Z   = (inversefunc(pwm_2_thr,y_values = PID_Z/9.8) - 1500)  /(0.8) + 12.064 + 500
	    PID_pwm_Yaw = (inversefunc(pwm_2_thr,y_values = PID_Yaw/9.8) - 1500)/(0.4) + 12.064

	    PID[0] = pwm_to_thrust(inversefunc(pwm_2_thr,y_values = PID_X/9.8))*9.8
	    PID[1] = pwm_to_thrust(inversefunc(pwm_2_thr,y_values = PID_Y/9.8))*9.8
	    PID[2] = pwm_to_thrust(inversefunc(pwm_2_thr,y_values = PID_Z/9.8))*9.8
	    PID[5] = pwm_to_thrust(inversefunc(pwm_2_thr,y_values = PID_Yaw/9.8))*9.8

	    master.mav.manual_control_send(master.target_system, PID_pwm_X, PID_pwm_Y, PID_pwm_Z, PID_pwm_Yaw, 0) # Thruster 
	    time.sleep(0.2)

	    Mrb = np.array([[   m,      0,      0,      0,   m*zg,      0],
	                    [   0,      m,      0,  -m*zg,      0,      0],
	                    [   0,      0,      m,      0,      0,      0],
	                    [   0,  -m*zg,      0,     Ix,      0,      0],
	                    [m*zg,      0,      0,      0,     Iy,      0],
	                    [   0,      0,      0,      0,      0,     Iz] ])

	    Ma  = - np.array([ [Xud,   0,   0,   0,   0,   0],
	                       [  0, Yvd,   0,   0,   0,   0],
	                       [  0,   0, Zwd,   0,   0,   0],
	                       [  0,   0,   0, Kpd,   0,   0],
	                       [  0,   0,   0,   0, Mqd,   0],
	                       [  0,   0,   0,   0,   0, Nrd] ])

	    M = Mrb + Ma

	    Crb = np.array ([ [   0,     0,     0,       0,     m*r,    -m*q],
	                      [   0,     0,     0,    -m*r,       0,     m*p],
	                      [   0,     0,     0,     m*q,    -m*p,       0],
	                      [   0,   m*r,  -m*q,       0,    Iz*r,   -Iy*q],
	                      [-m*r,     0,   m*p,   -Iz*r,       0,    Ix*p],
	                      [ m*q,  -m*p,     0,    Iy*q,   -Ix*p,       0] ])

	    Ca = np.array([ [     0,       0,      0,       0,   -Zwd*w,  Yvd*v],
	                    [     0,       0,      0,   Zwd*w,        0, -Xud*u],
	                    [     0,       0,      0,  -Yvd*v,    Xud*u,      0],
	                    [     0,  -Zwd*w,  Yvd*v,       0,   -Nrd*r,  Mqd*q],
	                    [ Zwd*w,       0, -Xud*u,   Nrd*r,        0, -Kpd*p],
	                    [-Yvd*v,   Xud*u,      0,  -Mqd*q,    Kpd*p,      0] ])
	    
	    C = Crb+Ca

	    D = - np.array([ [ Xu + Xuu*abs(u),                0,                 0,                 0,                 0,                 0],
	                     [               0,  Yv + Yvv*abs(v),                 0,                 0,                 0,                 0],
	                     [               0,                0,   Zw + Zww*abs(w),                 0,                 0,                 0],
	                     [               0,                0,                 0,   Kp + Kpp*abs(p),                 0,                 0],
	                     [               0,                0,                 0,                 0,   Mq + Mqq*abs(q),                 0],
	                     [               0,                0,                 0,                 0,                 0,   Nr + Nrr*abs(r)] ])
	    
	    g = np.array([ [     (B-W)*(2*E1*E3 - 2*E2*qj)],
	                   [     (B-W)*(2*E2*E3 - 2*E1*qj)],
	                   [ (W-B)*(2*E1**2 + 2*E2**2 - 1)],
	                   [      zg*W*(2*E2*E3 + 2*E1*qj)],
	                   [      zg*W*(2*E1*E3 - 2*E2*qj)],
	                   [                           0] ]) 

	    def func(t,X,V):
	      return np.dot( np.linalg.inv(M), (PID - np.dot(C,V) - np.dot(D,V) - g) )
	      #return np.dot( np.linalg.inv(M), (PID - np.dot(C,V) - np.dot(D,V)) )

	    X = np.array([ [x], [y], [z], [roll], [pitch], [yaw] ])
	    V = np.array([ [u], [v], [w],    [p],     [q],   [r] ])

	    Xq = np.array([ [x], [y], [z], e2q(X[3:6])[0], e2q(X[3:6])[1], e2q(X[3:6])[2], e2q(X[3:6])[3]   ])
	    Vq = np.array([ [u], [v], [w], e2q(V[3:6])[0], e2q(V[3:6])[1], e2q(V[3:6])[2], e2q(V[3:6])[3]   ])

	    ## X,V [World -> Body] 
	    Xb = np.dot(np.linalg.pinv(J), Xq)
	    Vb = np.dot(np.linalg.pinv(J), Vq)
	    
	    ## Runge-Kutta
	    kx1 = Vb
	    kv1 = func( t, Xb, Vb )

	    kx2 = Vb + h*kv1/2
	    kv2 = func( t + h/2, Xb + h*kx1/2, Vb + h*kv1/2 )

	    kx3 = Vb + h*kv2/2
	    kv3 = func( t + h/2, Xb + h*kx2/2, Vb + h*kv2/2 )

	    kx4 = Vb + h*kv3
	    kv4 = func( t + h, Xb + h*kx3, Vb + h*kv3 )

	    dx = h*(kx1 + 2*kx2 + 2*kx3 + kx4)/6
	    dv = h*(kv1 + 2*kv2 + 2*kv3 + kv4)/6
	    
	    Xb = Xb + dx

	    Vb = Vb + dv

	    ## X,V [Body -> World]  
	    Xnq = np.dot(J, Xb)
	    Vnq = np.dot(J, Vb)

	    Xn = np.array([ [float(Xnq[0])], [float(Xnq[1])], [float(Xnq[2])],  [0], [0], [0] ])
	    Vn = np.array([ [float(Vnq[0])], [float(Vnq[1])], [float(Vnq[2])],  [0], [0], [0] ])

	    Q2EX = q2e(Xnq[3:7])
	    Xn[3] = float(Q2EX[0])
	    Xn[4] = float(Q2EX[1])
	    Xn[5] = float(Q2EX[2])

	    Q2EV = q2e(Vnq[3:7])
	    Vn[3] = float(Q2EV[0])
	    Vn[4] = float(Q2EV[1])
	    Vn[5] = float(Q2EV[2])
	 
	    I = np.vstack([ Xn, Vn ])

	    print('X:',I[0])
	    print('Y:',I[1])

	    E = ( (float(DP[0]) - float(I[0]))**2 + (float(DP[1]) - float(I[1]))**2+ (float(DP[2]) - float(I[2]))**2)**(0.5)
	    E_p = ( (float(DP[0]) - float(I[0]))**2 + (float(DP[1]) - float(I[1]))**2 )**(0.5)
	    E_v = ( (float(DP[6]) - float(I[6]))**2 + (float(DP[7]) - float(I[7]))**2 )**(0.5)
	    
	    print('Error:',E_p)

	    if 0.01 > abs(E_p):
	      break

	    errorPrevious = error

	    st = time.time() - start_time

	    print(st)

	    t = t + h

def get_imu_data(master):
	return master.recv_match(type='SCALED_IMU2', blocking=True)

def get_imu_matrix(master):
	msg = get_imu_data(master)
	return [msg.xacc, msg.yacc. msg.zacc, msg.xgyro, msg.ygyro, msg.zgyro]

dt = 1.8*(0.14)

def rotation(Yaw):

	## IMU initial value
	imu_vel_x = 0
	imu_vel_y = 0
	imu_vel_z = 0

	imu_pos_x = 0
	imu_pos_y = 0
	imu_pos_z = 0

	imu_ang_r = 0
	imu_ang_p = 0
	imu_ang_y = 0

	while(1):

		start_time = time.time()

		imu_string = str(get_imu_data(master_1))
		imu_list = imu_string.split(',')

		#imu_xacc = float(str(imu_list[1]).split(':')[1])*9.8/1000 # m/s**2
		#imu_yacc = float(str(imu_list[2]).split(':')[1])*9.8/1000 # m/s**2
		#imu_zacc = float(str(imu_list[3]).split(':')[1])*9.8/1000 + 9.8 # m/s**2

		#imu_xgyro = float(str(imu_list[4]).split(':')[1])/1000 # rad/sec
		#imu_ygyro = float(str(imu_list[5]).split(':')[1])/1000 # rad/sec
		imu_zgyro = float(str(imu_list[6]).split(':')[1])/1000 # rad/sec 

		#imu_vel_x = imu_vel_x + imu_xacc * dt # m/s
		#imu_vel_y = imu_vel_y + imu_yacc * dt # m/s
		#imu_vel_z = imu_vel_z + imu_zacc * dt # m/s

		#imu_pos_x = imu_pos_x + imu_vel_x * dt # m
		#imu_pos_y = imu_pos_y + imu_vel_y * dt # m
		#imu_pos_z = imu_pos_z + imu_vel_z * dt # m

		#imu_ang_r = imu_ang_r + imu_xgyro * dt # rad
		#imu_ang_p = imu_ang_p + imu_ygyro * dt # rad
		imu_ang_y = imu_ang_y + imu_zgyro * dt # rad

		print('zgyro:',imu_zgyro)
		print('Yaw:',imu_ang_y)

		print(abs(abs(imu_ang_y) - abs(Yaw)))

		if abs(abs(imu_ang_y) - abs(Yaw)) < 0.03 :
			if Yaw < 0:
				master_1.mav.manual_control_send(master_1.target_system, 0, 0, 0, (Yaw/abs(Yaw))*600*0.82, 0)
				time.sleep(3)
				master_1.mav.manual_control_send(master_1.target_system, 0, 0, 0, 0, 0)
			break
		else:
			#master_1.mav.manual_control_send(master_1.target_system, 0, 0, 0, (Yaw/abs(Yaw))*600, 0)
			
			if Yaw < 0:
				master_1.mav.manual_control_send(master_1.target_system, 0, 0, 0, (Yaw/abs(Yaw))*600*0.82, 0)
			else:
				master_1.mav.manual_control_send(master_1.target_system, 0, 0, 0, (Yaw/abs(Yaw))*600*0.9, 0)
			
		
		st = time.time() - start_time

		print('sampling time:',st)

if __name__ =='__main__':

	## connect
	master_1 = mavutil.mavlink_connection('udpin:0.0.0.0:14560')

	master_1.wait_heartbeat()

	## Arm
	master_1.mav.command_long_send(master_1.target_system, master_1.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
	'''
	rotation(math.pi/2)
	go_point(master_1, 1, 0, 0, 0)
	rotation(-math.pi/2)

	while(1):
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		rotation(-math.pi/2)

		go_point(master_1, 1, 0, 0, 0)
		rotation(-math.pi/2)

		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)

		rotation(-math.pi/2)

		go_point(master_1, 1, 0, 0, 0)
		rotation(-math.pi/2)
	'''
	#rotation(math.pi/2)
	while(1):
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		rotation(math.pi/2)
		go_point(master_1, 1, 0, 0, 0)
		rotation(math.pi/2)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 1, 0, 0, 0)
		go_point(master_1, 0.5, 0, 0, 0)
		rotation(math.pi/2)
		go_point(master_1, 1, 0, 0, 0)
		rotation(math.pi/2)

	'''
	go_point(master_1, 1, 0, 0, 0)
	go_point(master_1, 1, 0, 0, 0)
	rotation( math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	rotation(-math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	go_point(master_1, 1, 0, 0, 0)
	rotation(-math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	rotation(-math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	go_point(master_1, 1, 0, 0, 0)
	rotation(-math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	rotation( math.pi/2)

	go_point(master_1, 1, 0, 0, 0)
	go_point(master_1, 1, 0, 0, 0)
	'''