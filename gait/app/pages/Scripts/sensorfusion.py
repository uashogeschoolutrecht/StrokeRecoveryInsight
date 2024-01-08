import numpy as np
from . import commonFunctions
from . import quaternionfunction

class sensorFusion:
    '''
    This function is used to rotate the sensor from a local to the global axis. 
    First the raw acceleration and gyroscope data is combined into quaternions 
    using the Madgwick sensor obj.fusion algorithm. This gives us the orientation
    of the sensor in the global axis. This in turn can be used to rotate the 
    raw acceleration signal to generate linear acceleration. By integrating twice 
    position can be calculated. This enables us to calculate the forward 
    displacement using a zero verlocity update
    '''
    def __init__(self, obj):
            '''
            Create a sensorfusion object with quaternions
            '''
            self.q = [1,0,0,0]
            self.ki = 0                                     # Enables to correct the signal at the start
            self.kpinit = 200                               # Enables to correct the signal at the start
            self.initperiod = 5                             #    Enables to correct the signal at the start
            self.interror= np.array([0.0, 0.0, 0.0])              # Error of the gyroscope
            self.sampleperiod = obj.sampleFreq
            self.kpramped = 0
            self.beta = 1
            
    def LinearAcceleration(obj, plotje = None):
        '''
        Calculate the acceleration in a global axis 
        '''
        
        accconv         = obj.acceleration[0:obj.walkSig[0]]  
        if (len(accconv) > 100)    :                                         
            for i in range(len(accconv)):                                             # Initiallise the quartonion, this needs a moment to set in
                obj.fusion.update_6dof([np.mean(accconv[:,0]), 
                                        np.mean(accconv[:,1]), 
                                        np.mean(accconv[:,2])],
                                        [0,0,0], kp= 1 )
         
        quaternion          = np.zeros((len(obj.acceleration),4))
        linearacc           = np.zeros((len(obj.acceleration),3))
        

        tmp_acceleration    = np.array(obj.acceleration)
        tmp_gyroscope       = np.array(obj.gyroscope)
        
        standphases = obj.standphases
        stationary = np.array([])
        for i in standphases:
            stationary = np.concatenate((stationary,i))
            
        for i in range(len(obj.acceleration)):                                        # calculate quaternion
            if i in stationary:
                quaternion[i] = obj.fusion.update_6dof(tmp_acceleration[i], 
                           tmp_gyroscope[i], kp = 0.5)
            else:
                quaternion[i] = obj.fusion.update_6dof(tmp_acceleration[i], 
                   tmp_gyroscope[i], kp = 0)
            linearacc[i] = obj.fusion.linearacc(obj.acceleration[i], quaternion[i])
        
        obj.stationary = stationary
        
        linearacc            *= 9.81     
        linearacc[:,2]      -= 9.81     
                                           # Calculate acceleration  
        if plotje:
            commonFunctions.plot1(linearacc, title = 'Corrected acceleration signals AP, ML, VT'
                             , xlabel = 'Samples', ylabel = 'Acceleration [m/s^2]')
        return linearacc, quaternion
    
    def VelocityPosition(obj, rotate = None, printje = None,
                         veldriftcorrection = None):
        '''
        Integrate the acceleration twice to achieve velocity and position.
        For every stride we calculate the velocity drift. We added the 
        possibility to rotate the signal that the ML direction is zero 
        over time. 
        '''
        standphases = obj.standphases
        stationary = np.array([])
        for i in standphases:
            stationary = np.concatenate((stationary,i))
        obj.stationary = stationary    
        
        velocity            = obj.fusion.velocity(obj.globalacceleration, 
                                                  obj.stationary
                                              )                  # Input: 3D linear acceleration & standphases. Output: Velocity        
        if veldriftcorrection:
            veldrift = np.zeros((len(velocity),3))
            for count, value in enumerate(obj.standphases[1:-1]):
                driftrate = velocity[value[0]-1] / (value[0] - obj.standphases[count][-1])
                enum = np.arange(value[0] - obj.standphases[count][-1])
                drift = np.transpose(np.array((enum,enum,enum))) * driftrate
                veldrift[obj.standphases[count][-1]:value[0]] = drift
            velocity            -= veldrift
            
        position            = obj.fusion.position(velocity)     
        


        if rotate:
            x_tmp               = position[-1]                                          # Set the displacement to a global axis
            z                   = np.array([0, 0, 1])     
            y                   = np.cross(z,x_tmp)                      
            x                   = np.cross(y,z)
            z                   = z / np.linalg.norm(z)
            y                   = y / np.linalg.norm(y)
            x                   = x / np.linalg.norm(x)
            R_glob              = np.array([x, y, z])
            forward_velocity    = commonFunctions.rotsig(velocity, R_glob)
            displacement        = commonFunctions.rotsig(position, R_glob)
            totdist             = np.sum(np.abs(np.diff(displacement[:,0]) ))
            print('distance walked: ',totdist, 'm')
            
          
            return forward_velocity, displacement
        else:
            #print('distance walked: ',totdist, 'm')
            return velocity, position
        
            
    def update_6dof(self, acceleration, gyroscope, kp, quaternion = None):   
        '''
        the sensor fusion algoritm based on the method of Madgwick.
        An efficient orientation filter for inertial and inertial/magnetic sensor arrays
        Sebastian O.H. Madgwick April 30, 2010
        This script combines the gyroscope and accelerometer to calculate the
        orientation of the sensor in a global axis. 
        '''
        if quaternion:
            self.q = quaternion
        self.kp = kp
        
        gx, gy, gz = gyroscope
        acceleration /=  np.linalg.norm(acceleration)

        # Compute error between estimated and measured directoin of gravity
        v = np.array([2*(self.q[1]*self.q[3] - self.q[0]*self.q[2]), 
                      2*(self.q[0]*self.q[1] + self.q[2]*self.q[3]), 
                      self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2]);               	

        # estimated direction of gravity -> What the accelerometer measures vs 
        # what the quaternion actual value is

        error = np.array(np.cross(v, acceleration))
        
        # Compute ramped Kp value used during init period
        if self.kpramped > self.kp:
            self.interror = np.array([0.0, 0.0, 0.0])
            self.kpramped -= ( (self.kpinit - self.kp) / (self.initperiod / self.sampleperiod))
        else: 
            self.kpramped = self.kp
            self.interror += error
            
        ref = gyroscope - (self.kp*error + self.ki*self.interror)                      # Apply feedback terms
                    # Apply feedback terms
        


        # Compute rate of change for quaternion
        qDot1 = 0.5 * np.array(quaternionfunction.q_mult(tuple(self.q), (0.0, ref[0],ref[1],ref[2])))
        
        # Integrate to yiel quaternion
        self.q = self.q + qDot1 * self.sampleperiod
        

        # Normalise quaternion
        self.q /= np.linalg.norm(self.q)
    
        # Store conjugate

        
        return quaternionfunction.q_conjugate(self.q)
        
    
    def update_9dof(self, acceleration, gyroscope, magnetometer, kp, quaternion = None):    
        '''
        the sensor fusion algoritm based on the method of Madgwick.
        An efficient orientation filter for inertial and inertial/magnetic sensor arrays
        Sebastian O.H. Madgwick April 30, 2010
        This script combines the gyroscope, accelerometer and magnetometer to calculate the
        orientation of the sensor in a global axis. 
        '''
        mx, my, mz = magnetometer
        ax, ay, az = acceleration                  # Units irrelevant (normalised)
        gx, gy, gz = gyroscope  
        q1, q2, q3, q4 = self.q
        
        _2q1 = 2 * q1
        _2q2 = 2 * q2
        _2q3 = 2 * q3
        _2q4 = 2 * q4
        _2q1q3 = 2 * q1 * q3
        _2q3q4 = 2 * q3 * q4
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        # Normalise accelerometer measurement
        norm = np.sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0):
            return # handle NaN
        norm = 1 / norm                     # use reciprocal for division
        ax *= norm
        ay *= norm
        az *= norm

        # Normalise magnetometer measurement
        norm = np.sqrt(mx * mx + my * my + mz * mz)
        if (norm == 0):
            return                          # handle NaN
        norm = 1 / norm                     # use reciprocal for division
        mx *= norm
        my *= norm
        mz *= norm

        # Reference direction of Earth's magnetic field
        _2q1mx = 2 * q1 * mx
        _2q1my = 2 * q1 * my
        _2q1mz = 2 * q1 * mz
        _2q2mx = 2 * q2 * mx
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2 * _2bx
        _4bz = 2 * _2bz

        # Gradient descent algorithm corrective step
        s1 = (-_2q3 * (2 * q2q4 - _2q1q3 - ax) + _2q2 * (2 * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4)
             + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
             + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s2 = (_2q4 * (2 * q2q4 - _2q1q3 - ax) + _2q1 * (2 * q1q2 + _2q3q4 - ay) - 4 * q2 * (1 - 2 * q2q2 - 2 * q3q3 - az)
             + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4)
             + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s3 = (-_2q1 * (2 * q2q4 - _2q1q3 - ax) + _2q4 * (2 * q1q2 + _2q3q4 - ay) - 4 * q3 * (1 - 2 * q2q2 - 2 * q3q3 - az)
             + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
             + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
             + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s4 = (_2q2 * (2 * q2q4 - _2q1q3 - ax) + _2q3 * (2 * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4)
              + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
              + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        norm = 1 / np.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    # normalise step magnitude
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate to yield quaternion
        deltat = self.sampleperiod
        q1 += qDot1 * deltat
        q2 += qDot2 * deltat
        q3 += qDot3 * deltat
        q4 += qDot4 * deltat
        norm = 1 / np.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)    # normalise quaternion
        self.q = q1 * norm, q2 * norm, q3 * norm, q4 * norm
        
        
        return quaternionfunction.q_conjugate(self.q)

        
    def linearacc(self, acceleration, quaternion):
        '''
        Convert sensor acceration to global acceleration using quaternions.
        '''
        v1 = (0.0, acceleration[0], acceleration[1], acceleration[2])
        q1 = quaternionfunction.q_conjugate(quaternion)

        return quaternionfunction.qv_mult(v1, q1)
    
    def velocity(self, linearacc, stationary):
        '''
        Integrate acceleration to velocity
        Includes the zero velocity update
        '''
        vel = np.zeros((len(linearacc),3))
        for i in range(1, len(vel)):
            j = i - 1
            vel[i] = vel[j] + linearacc[i] * self.sampleperiod
            if (i in stationary) :
                vel[i] = 0

        return vel
        
    def position(self, vel):
        '''
        Integrate velocity to position
        '''
        pos = np.zeros((len(vel),3))
        
        for i in range(1, len(pos)):
            j = i - 1
            pos[i] = pos[j] + vel[i] * self.sampleperiod 
        return pos
    
    def caclrot(self, quaternion, order):
        '''
        Calculate the orientation of the sensor in a global axis. 
        '''
        from scipy.spatial.transform import Rotation as R
        self.q = quaternion
        r = R.from_quat([self.q[0], self.q[1], self.q[2], self.q[3]])
        ang = r.as_euler(order, degrees=True)
        
        return ang
    
    def spatTempoutcome(obj, devideInparts, printje = None, plotje = None):
        '''
        In this function the spatial measures are calculated. 
        Direction: 'AP, 'ML', 'VT'
        We calculate: 
        Mean, std and rms stridelength in AP direction
        Mean, std and rms stridevelocity in AP direction
        '''
        position = obj.position
        velocity = obj.velocity
        
        forwardvelocityx = np.abs(np.diff(velocity[:,0]))
        forwardvelocityy = np.abs(np.diff(velocity[:,1]))
        forwardvelocity = np.hypot(forwardvelocityx,forwardvelocityy)
        
        forwarddisplacementx = np.abs(np.diff(position[:,0]))
        forwarddisplacementy = np.abs(np.diff(position[:,1]))
        forwarddisplacement = np.hypot(np.cumsum(forwarddisplacementx),np.cumsum(forwarddisplacementy))
        
        if plotje:
            commonFunctions.plot2(forwardvelocity, forwarddisplacement,
                                 title = 'Forward velocity and position',
                                 xlabel = 'Time in seconds', ylabel1 = 'Velocity [m/s]',
                                 ylabel2 = 'Position [m]'
                                 )      
        strideDist                      = np.zeros(obj.stridenum)
        strideVel                       = np.zeros(obj.stridenum)
        standphases                     = obj.standphases
        
        for count, value in enumerate(standphases[:-1]):
            strideDist[count]           = np.abs(np.diff((forwarddisplacement[value[-1]],forwarddisplacement[standphases[count+1][0]])))[0]
            strideVel[count]            = np.max(velocity[value[-1]:standphases[count+1][0]])
        
        strideDist = strideDist[np.where(strideDist < 2* np.mean(strideDist))[0]]
        strideVel = strideVel[np.where(strideVel < 2* np.mean(strideVel))[0]]
        
        forwarddisplacement             = np.sum(strideDist)
        obj.kmph                        = (np.max(forwarddisplacement ) * 30) / 1000

        
        meanStrideDistperstep = []
        stdStrideDistperstep= []
        meanstrideVelperstepperstep= []
        stdstrideVelperstepperstep= []
            
        count = int(np.floor(len(strideDist[5:-5]) / devideInparts))
        for i in range(count):
            tmpValPos = strideDist[devideInparts*i+5:devideInparts*i+5 + devideInparts]
            meanStrideDistperstep.append(np.mean(tmpValPos))
            stdStrideDistperstep.append(np.std(tmpValPos))
            tmpValVel = strideVel[devideInparts*i+5:devideInparts*i+5 + devideInparts]
            meanstrideVelperstepperstep.append(np.mean(tmpValVel))
            stdstrideVelperstepperstep.append(np.std(tmpValVel))
            
        obj.totdist                     =   np.max(forwarddisplacement )
        obj.meanStrideDistperstep       = np.mean(meanStrideDistperstep)
        obj.stdStrideDistperstep        = np.mean(stdStrideDistperstep)
        obj.meanstrideVelperstepperstep = np.mean(meanstrideVelperstepperstep)
        obj.stdstrideVelperstepperstep  = np.mean(stdstrideVelperstepperstep)
        
        
        if printje:
            print('Distance walked: ',  np.max(forwarddisplacement ), 'm')
            print('Mean distance per stride: ',  obj.meanStrideDistperstep)
            print('STD distance per stride: ',  obj.stdStrideDistperstep)
            print('Mean velocity per stride: ',  obj.meanstrideVelperstepperstep )
            print('STD velocity per stride: ',  obj.stdstrideVelperstepperstep )
            print('Kilometer per hour: ',  obj.kmph)





        
        