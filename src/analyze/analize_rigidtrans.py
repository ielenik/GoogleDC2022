from collections import deque
from bisect import insort, bisect_left
from itertools import islice
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import math
import random
from ..utils.magnet_model import Magnetometer

from tensorflow.python.framework import function

@tf.custom_gradient
def my_norm(x):
    y = tf.linalg.norm(x, axis = -1)
    def grad(dy):
        return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

    return y, grad

def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result


def showIntegrate(values, times, truth):
    start = times[0]
    valint = []
    curval = np.zeros((3))
    curvalcount = 0
    for ind in range(len(values)):
        if times[ind] > start + 1000:
            if curvalcount > 0:
                valint.append(curval / curvalcount)
            curval = np.zeros((3))
            curvalcount = 0
            start = times[ind]
        curvalcount += 1
        curval += values[ind]

    valint = np.array(valint)
    legend = [ 'X sens', 'Y sens', 'Z sens',]
    for i in range(3):
        pyplot.plot( np.arange(len(valint)), valint[:,i])

    if len(truth.shape) == 2:
        for i in range(truth.shape[1]):
            pyplot.plot( np.arange(len(truth)),     truth[:,i])
            legend.append('truth ' + str(i))
    else:
        pyplot.plot( np.arange(len(truth)),     truth)
        legend.append('truth')
    pyplot.legend(legend)
    pyplot.show()        

def mult_np(a, b):
    return np.array([a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],  
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],  
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]])

def mult(a, b):
    return tf.stack([a[:,3] * b[:,0] + a[:,0] * b[:,3] + a[:,1] * b[:,2] - a[:,2] * b[:,1],  
        a[:,3] * b[:,1] - a[:,0] * b[:,2] + a[:,1] * b[:,3] + a[:,2] * b[:,0],  
        a[:,3] * b[:,2] + a[:,0] * b[:,1] - a[:,1] * b[:,0] + a[:,2] * b[:,3],
        a[:,3] * b[:,3] - a[:,0] * b[:,0] - a[:,1] * b[:,1] - a[:,2] * b[:,2]], axis = 1)

def inv(a):
    return tf.stack([-a[:,0], -a[:,1], -a[:,2], a[:,3]], axis = 1)

def inv_np(a):
    return [-a[0], -a[1], -a[2], a[3]]

def to_quat(omega, dt):
    omegaMagnitude = np.linalg.norm(omega)
    if (omegaMagnitude < 0.00001):
        omegaMagnitude = 1

    thetaOverTwo = omegaMagnitude * dt / 2.0
    sinThetaOverTwo = math.sin(thetaOverTwo) / omegaMagnitude
    cosThetaOverTwo = math.cos(thetaOverTwo) 

    return np.array([sinThetaOverTwo * omega[0], sinThetaOverTwo * omega[1], sinThetaOverTwo * omega[2], cosThetaOverTwo])
def get_quat(v1,v2):
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    res = np.zeros(4)
    res[:3] = np.cross(v1,v2)
    res[3] = 1 + np.dot(v1, v2)
    res = res/np.linalg.norm(res)
    return res

def transform(value, rotation):
    num12 = rotation[:,0] + rotation[:,0]
    num2 = rotation[:,1] + rotation[:,1]
    num = rotation[:,2] + rotation[:,2]
    num11 = rotation[:,3] * num12
    num10 = rotation[:,3] * num2
    num9 = rotation[:,3] * num
    num8 = rotation[:,0] * num12
    num7 = rotation[:,0] * num2
    num6 = rotation[:,0] * num
    num5 = rotation[:,1] * num2
    num4 = rotation[:,1] * num
    num3 = rotation[:,2] * num
    num15 = ((value[:,0] * ((1. - num5) - num3)) + (value[:,1] * (num7 - num9))) + (value[:,2] * (num6 + num10))
    num14 = ((value[:,0] * (num7 + num9)) + (value[:,1] * ((1. - num8) - num3))) + (value[:,2] * (num4 - num11))
    num13 = ((value[:,0] * (num6 - num10)) + (value[:,1] * (num4 + num11))) + (value[:,2] * ((1. - num8) - num5))
    return tf.stack([num15,num14,num13], axis = 1)


def transform_np(value, rotation):
    num12 =rotation[0] + rotation[0]
    num2 = rotation[1] + rotation[1]
    num =  rotation[2] + rotation[2]
    num11 =rotation[3] * num12
    num10 =rotation[3] * num2
    num9 = rotation[3] * num
    num8 = rotation[0] * num12
    num7 = rotation[0] * num2
    num6 = rotation[0] * num
    num5 = rotation[1] * num2
    num4 = rotation[1] * num
    num3 = rotation[2] * num
    num15 = ((value[0] * ((1. - num5) - num3)) + (value[1] * (num7 - num9))) + (value[2] * (num6 + num10))
    num14 = ((value[0] * (num7 + num9)) + (value[1] * ((1. - num8) - num3))) + (value[2] * (num4 - num11))
    num13 = ((value[0] * (num6 - num10)) + (value[1] * (num4 + num11))) + (value[2] * ((1. - num8) - num5))
    return np.array([num15,num14,num13])


class MagnetCallibration(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagnetCallibration, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MagnetCallibration, self).build(input_shape)

        self.hard_iron = self.add_weight(name='hard_iron', shape=(1, 3), trainable=True, initializer=tf.keras.initializers.Constant(value=[[1,1,1]]))
        self.soft_iron = self.add_weight(name='soft_iron', shape=(6), trainable=True, initializer=tf.keras.initializers.Constant(value=[1,1,1, 0,0,0]))
    def call(self, mag_input):
        mag_input = mag_input - self.hard_iron
        
        magx = mag_input[:,0]*mag_input[:,0]*self.soft_iron[0] + mag_input[:,0]*mag_input[:,1]*self.soft_iron[3] + mag_input[:,0]*mag_input[:,2]*self.soft_iron[4]
        magy = mag_input[:,1]*mag_input[:,0]*self.soft_iron[3] + mag_input[:,1]*mag_input[:,1]*self.soft_iron[1] + mag_input[:,0]*mag_input[:,2]*self.soft_iron[5]
        magz = mag_input[:,2]*mag_input[:,0]*self.soft_iron[4] + mag_input[:,2]*mag_input[:,1]*self.soft_iron[5] + mag_input[:,2]*mag_input[:,2]*self.soft_iron[2]

        self.add_loss(tf.abs(1-magx - magy - magz))
        return 1-magx - magy - magz


class RigidModelGroundTruth(tf.keras.layers.Layer):
    def __init__(self, poses, dires, mags, **kwargs):
        super(RigidModelGroundTruth, self).__init__(**kwargs)
        self.epochs = len(poses)
        self.poses = np.array(poses)
        self.dires = np.array(dires)

        self.down  = tf.Variable([[0.,0.,-1.]], trainable=False, dtype = tf.float32, name = 'down')
        self.fwd  = tf.Variable([[0.,1.,0.]], trainable=False, dtype = tf.float32, name = 'fwd')
        
        self.positions_start = tf.Variable(self.poses, trainable=False, dtype = tf.float32, name = 'positions_start')

        self.positions = tf.Variable(self.poses, trainable=True, dtype = tf.float32, name = 'positions')
        self.positions_fourier = tf.Variable(np.zeros((1, self.epochs, 3)), trainable=True, dtype = tf.float32, name = 'positions_fourier')
        
        #self.speed = tf.Variable(np.zeros((self.poses[1:] - self.poses[:-1]).shape), trainable=True, dtype = tf.float32, name = 'speed')
        self.idquat = tf.Variable([[0.,0.,0.,1.]], trainable=False, dtype = tf.float32, name = 'idquat')

        quats = np.zeros((self.dires.shape[0]+5,4))
        quats[:,3] = 1
        fwd = np.array([0.,1.,0])
        self.orient_init = get_quat([0,1,0], [0,0,1])
        if False:
            for i in range(self.dires.shape[0]):
                quats[i] = get_quat(fwd, self.dires[i])
        else:
            for i in range(mags.shape[0]):
                mag = transform_np(mags[i],self.orient_init)
                quats[i] = get_quat(mag, fwd)
        #self.orient_init = get_quat([0,1,0], [0,0,1])
        self.quat = tf.Variable(quats, trainable=True, dtype = tf.float32, name = 'quat')


    def build(self, input_shape):
        super(RigidModelGroundTruth, self).build(input_shape)

        self.g = self.add_weight(name='g', shape=(1, 1), trainable=True, initializer=tf.keras.initializers.Constant(value=9.81))
        self.orientation = self.add_weight(name='orientation', shape=(1,4), trainable=True, initializer=tf.keras.initializers.Constant(value=[self.orient_init]))
        #self.mag_orientation = self.add_weight(name='mag_orientation', shape=(1,4), trainable=True, initializer=tf.keras.initializers.Constant(value=[self.orient_init]))
        self.north = self.add_weight(name='north', shape=(1,3), trainable=False, initializer=tf.keras.initializers.Constant(value=[[0,1,0]]))
        self.ort_vector = self.add_weight(name='ort_vector', shape=(1,3), trainable=False, initializer=tf.keras.initializers.Constant(value=[[0,1,0]]))
        self.dwn_grav = self.add_weight(name='dwn_grav', shape=(1,3), trainable=False, initializer=tf.keras.initializers.Constant(value=[[0,-1,0]]))

    def get_position_correction(self,index):
        indicies = tf.cast(tf.reshape(index,[-1]), tf.float32)
        xval = indicies/self.epochs*math.pi
        xval = tf.reshape(xval,[-1,1])
        perd = tf.cast(tf.range(self.epochs), tf.float32)
        perd = tf.reshape(perd, [1,-1])
        args = xval*perd
        return tf.reduce_sum(self.positions_fourier*tf.expand_dims(tf.math.cos(args), -1), axis = 1)
        
    def call(self, inputs):

        epoch_input, acs_input, gyr_input, mag_input = inputs

        ort = mag_input

        indicies = tf.reshape(epoch_input,[-1])
        qt1, _ = tf.linalg.normalize(tf.gather(self.quat, indicies, axis = 0), axis = -1)
        qt2, _ = tf.linalg.normalize(tf.gather(self.quat, indicies - 1, axis = 0), axis = -1)
        orientation, _ = tf.linalg.normalize(self.orientation, axis = -1)
        #mag_orientation, _ = tf.linalg.normalize(self.mag_orientation, axis = -1)
        
        pt1 = tf.cast(indicies, tf.float32)/self.epochs*math.pi
        pt2 = pt1 - 1./self.epochs*math.pi

        posS = tf.gather(self.positions_start, indicies, axis = 0)
        pos0 = tf.gather(self.positions, indicies, axis = 0) + self.get_position_correction(indicies)
        pos1 = tf.gather(self.positions, indicies+1, axis = 0) + self.get_position_correction(indicies+1)
        pos2 = tf.gather(self.positions, indicies-1, axis = 0) + self.get_position_correction(indicies-1)

        speed1 = pos1 - pos0
        speed2 = pos0 - pos2

        north_pred = ort
#        north_pred = transform(ort,mult(qt1, orientation))
        #north_pred = transform(ort,np.array([[0,0,1,0]]))
        north_pred = transform(north_pred,orientation)
        north_pred = transform(north_pred,qt1)
        north, _ = tf.linalg.normalize(self.north,axis = -1)
        mag_loss = tf.reduce_mean(my_norm((north - north_pred)[:,:2]))

        
        acsr = speed1 - speed2
        acsi = transform(acs_input,orientation)
        acsi = transform(acsi, qt1)
        acsi = acsi + self.down*self.g
        #tf.print(tf.math.top_k(my_norm(acsr-acsi),4)[0])
        #ind_top = tf.math.top_k(my_norm(acsr-acsi),4)[1]
        #acs_loss = tf.reduce_mean(tf.square(acsi - acsr))
        acs_loss = tf.reduce_mean(my_norm(acsi - acsr))
        #tf.print(acs_loss)
        #tf.print(tf.gather(indicies,ind_top))

        speed_loss = tf.reduce_mean((my_norm(speed1) - tf.reduce_sum(speed1*transform(self.fwd,qt1), axis = -1))/(my_norm(speed1) + 0.01))
        #bnd_loss = tf.reduce_mean(tf.abs(self.speed[1]) + tf.abs(self.speed[-2]))

        #qt1 = mult(qt1, orientation)
        #qt2 = mult(qt2, orientation)
        #quat_loss = tf.reduce_mean(tf.norm(gyr_input - [0,0,0,1],axis=-1))
        iqt1 = inv(qt1)
        true_quat = mult(iqt1, qt2)
        local_gyr = mult(mult(inv(orientation), gyr_input), orientation)
        quat_loss = tf.reduce_mean(1 - tf.square(tf.reduce_sum(true_quat*local_gyr, axis = -1)))*10
        #quat_loss = tf.reduce_mean(1 - tf.abs(mult(mult(qt2,mult(iqt2,qt1)), inv(qt1))[:,3]))
        
        ort_vector, _ = tf.linalg.normalize(self.ort_vector, axis = -1)
        up_axe_loss = tf.reduce_mean(tf.abs(tf.reduce_sum(ort_vector*ort, axis=-1)))

        dwn_grav, _ = tf.linalg.normalize(self.dwn_grav, axis = -1)
        garv_axe_loss = tf.reduce_mean(-tf.reduce_sum(dwn_grav*acs_input, axis=-1))
        

        #self.add_loss(quat_loss*10)
        self.add_loss(acs_loss)
        self.add_loss(mag_loss)
        #self.add_loss(bnd_loss)
        self.add_loss(speed_loss)
        self.add_loss(tf.reduce_mean(my_norm(pos0-posS))/100000)

        #self.add_loss(tf.reduce_mean(my_norm(speed1))/100)

        #self.add_loss(up_axe_loss)
        #self.add_loss(garv_axe_loss)

        self.add_metric(mag_loss,'mag')
        self.add_metric(quat_loss,'quat')
        self.add_metric(acs_loss,'acs')
        self.add_metric(speed_loss,'speed')
        self.add_metric(tf.reduce_mean(my_norm(pos0-posS))/100000,'mean')
        #self.add_metric(bnd_loss,'bnd')
        self.add_metric(self.g,'g')
        self.add_metric(up_axe_loss,'up_axe_loss')
        self.add_metric(garv_axe_loss,'garv_axe_loss')

        return [qt1, speed1, pos0 ]

    def compute_output_shape(self, _):
        return (1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'poses' : self.poses,
            'dires' : self.dires,
        })
        return config        

def rigidModelOnTruthData(acsvalues,acstimes,gyrvalues,gyrtimes,magvalues,magtimes, poses, course, epochtimes, initguess, path):


    beg_time = epochtimes[0]
    end_time = epochtimes[-1]
    nummesures = int((end_time-beg_time + 999)/1000)
    
    quat_epochs = np.zeros((nummesures,4))
    gyr_epochs = np.zeros((nummesures,3))
    acs_epochs = np.zeros((nummesures,3))
    mag_epochs = np.zeros((nummesures,3))
    mag_epochs[:,2] = 1
    quat_epochs[:,3] = 1


    acstimes += -315964800000 + 18000
    gyrtimes += -315964800000 + 18000
    magtimes += -315964800000 + 18000

    def accum_values(times,values,epochs):
        for i in range(1, len(times)):
            epoch = int((times[i] - beg_time)/1000)
            if epoch < 0 or epoch >= nummesures:
                continue
            dt = (times[i] - times[i-1])/1000
            epochs[epoch] += values[i]*dt

    accum_values(acstimes,acsvalues,acs_epochs)
    accum_values(magtimes,magvalues,mag_epochs)
    accum_values(gyrtimes,gyrvalues,gyr_epochs)

    for i in range(nummesures):
        quat_epochs[i] = to_quat(gyr_epochs[i], 1)

    mag_epochs -= np.mean(mag_epochs, axis = 0, keepdims = True)
    mag_epochs /= np.mean(np.linalg.norm(mag_epochs,axis=-1))

    mag = Magnetometer()
    mag.calibrate(mag_epochs)
    print()
    print(mag.A)
    print(mag.b)

    print(np.mean(np.abs(1- np.linalg.norm(mag_epochs,axis=-1))))
    mag_epochs = (mag_epochs - mag.b.reshape((1,3))).dot(mag.A)
    print(np.mean(np.abs(1- np.linalg.norm(mag_epochs,axis=-1))))



    legend = []
    
    orient_init = get_quat([0,1,0], [0,0,1])
    quats = np.zeros((course.shape[0]+5,4))
    quats[:,3] = 1
    fwd = np.array([0.,1.,0])
    for i in range(course.shape[0]):
        quats[i] = get_quat(fwd, course[i])
    
    '''
    quats_dif = quats[1:] - quats[:-1]
    for i in range(len(quats_dif)):
        quats_dif[i] = mult_np(inv_np(quats[i+1]), quats[i])
    for i in range(len(quat_epochs)):
        quat_epochs[i] = mult_np(mult_np(inv_np(orient_init), quat_epochs[i]),orient_init)
    
    for i in range(4):
        pyplot.plot( np.arange(len(quats_dif)),     quats_dif[:,i])
        legend.append('real ' + str(i))
        pyplot.plot( np.arange(len(quat_epochs)),    quat_epochs[:,i])
        legend.append('gyr ' + str(i))
    pyplot.legend(legend)
    pyplot.show()        
    
    legend = []
    acs_epochs2 = acs_epochs.copy()
    speed = poses[1:]-poses[:-1]

    acs = speed[1:]-speed[:-1]

    for i in range(len(acs)):
        acs_epochs2[i] = transform_np(acs_epochs2[i],orient_init)
        acs_epochs2[i] = transform_np(acs_epochs2[i],quats[i])
        acs_epochs2[i,2] -= 9.8

    print(np.mean(np.linalg.norm(acs_epochs2[:2888] - acs,axis = -1)))
    for i in range(3):
        pyplot.plot( np.arange(len(acs)),     acs[:,i])
        legend.append('real ' + str(i))
        pyplot.plot( np.arange(len(acs_epochs2)),     acs_epochs2[:,i])
        legend.append('gyro ' + str(i))
    pyplot.legend(legend)
    pyplot.show()        

    legend = []
    mag_epochs2 = mag_epochs.copy()

    for i in range(len(course)):
        mag_epochs2[i] = transform_np(mag_epochs2[i],orient_init)
        mag_epochs2[i] = transform_np(mag_epochs2[i],quats[i])

    print(np.mean(np.linalg.norm(mag_epochs2 - [[0,1,0]], axis = -1)))
    for i in range(3):
        pyplot.plot( np.arange(len(mag_epochs2)),     mag_epochs2[:,i])
        legend.append('mag ' + str(i))
    pyplot.legend(legend)
    pyplot.show()        
    '''

    print('***********\n', np.mean(np.square(poses[1:] - poses[:-1])))
    '''
    magnet_path = path+'/magnet_model.h5'
    try:
        magnet_calib = tf.keras.models.load_model(magnet_path)
    except:
        magnet_calib = MagnetCallibration()
        mag_input = tf.keras.layers.Input((3,), dtype=tf.float32, name = 'mag')
        mag_model = tf.keras.Model(mag_input, magnet_calib(mag_input))
        mag_model.compile(tf.keras.optimizers.SGD(learning_rate = 0.001, nesterov = True, momentum = 0.9))
        mag_model.summary()
        mag_model.fit(mag_epochs, np.zeros(len(mag_epochs)), epochs = 512,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 10,verbose=1,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(magnet_path,monitor='loss',verbose=1,save_best_only=True)]
                )

    print(magnet_calib.hard_iron.numpy())
    print(magnet_calib.soft_iron.numpy())
    '''
    if len(initguess) < len(poses):
        initguess = np.concatenate([initguess,poses[-(len(poses) - len(initguess)):]], axis = 0)

    rigid_model = RigidModelGroundTruth(
                     initguess,
                     #poses,
                     course, mag_epochs)


    epoch_input = tf.keras.layers.Input((1,), dtype=tf.int32, name = 'epoch')
    acs_input = tf.keras.layers.Input((3,), dtype=tf.float32, name = 'acs')
    gyr_input = tf.keras.layers.Input((4,), dtype=tf.float32, name = 'gyr')
    mag_input = tf.keras.layers.Input((3,), dtype=tf.float32, name = 'mag')

    allinput = [epoch_input,acs_input,gyr_input,mag_input]
    alloutput = rigid_model(allinput)

    imu_model = tf.keras.Model(allinput, alloutput)

    ds = tf.data.Dataset.from_tensor_slices({"epoch": np.arange(nummesures-4)+1, "acs": acs_epochs[:nummesures-4], "gyr": quat_epochs[:nummesures-4], "mag": mag_epochs[:nummesures-4]})
    ds = ds.repeat()
    ds = ds.shuffle(4096)
    ds = ds.batch(1024)

    # legend = []
    # mag_epochs[:,1] += 70
    # for i in range(3):
    #     pyplot.plot( np.arange(len(mag_epochs)),     mag_epochs[:,i])
    #     legend.append('src ' + str(i))
    # pyplot.legend(legend)
    # pyplot.show()

    if False:
        imu_model.load_weights('saveme.h5')
        imu_model.compile(tf.keras.optimizers.SGD(0.001, momentum = 0.9))
        imu_model.summary()

        imu_model.fit(ds, epochs = 1024, steps_per_epoch = 1024,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 5,verbose=1,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint("saveme.h5",monitor='loss',verbose=1,save_best_only=True)]
        )
    else:
        imu_model.load_weights('saveme.h5')

    ds = tf.data.Dataset.from_tensor_slices({"epoch": np.arange(nummesures-4)+1, "acs": acs_epochs[:-4], "gyr": quat_epochs[:-4], "mag": mag_epochs[:-4]})
    ds = ds.batch(1024)
    quats, speeds, restrack = imu_model.predict(ds)

    pyplot.plot(initguess[:,0],initguess[:,1]) 
    pyplot.plot(restrack[:,0],restrack[:,1]) 
    pyplot.plot(poses[:,0],poses[:,1]) 
    pyplot.show()

    acs1 = speeds[1:] - speeds[:-1]
    orient = rigid_model.orientation.numpy()[0]
    
    ort_vector = rigid_model.ort_vector.numpy()[0]
    north = rigid_model.north.numpy()[0]
    dwn_grav = rigid_model.dwn_grav.numpy()[0]
    dwn_grav /= np.linalg.norm(dwn_grav)
    orient /= np.linalg.norm(orient)
    north /= np.linalg.norm(north)
    ort_vector /= np.linalg.norm(ort_vector)
    quats /= np.linalg.norm(quats, axis = -1, keepdims=True)
    print(orient)
    g = rigid_model.g.numpy()[0][0]

    mag_epochs = mag_epochs[:-2]
    
    legend = []
    course2 = np.zeros((quats.shape[0],3))
    for i in range(quats.shape[0]):
        course2[i] = transform_np([0,1,0],quats[i])

    for i in range(3):
        pyplot.plot( np.arange(len(course)),     course[:,i])
        legend.append('true ' + str(i))
        pyplot.plot( np.arange(len(course2)),     course2[:,i])
        legend.append('pred ' + str(i))

    pyplot.legend(legend)
    pyplot.show()        


    legend = []
    speeds2 = poses[1:] - poses[:-1]
    for i in range(3):
        pyplot.plot( np.arange(len(speeds2)),     speeds2[:,i])
        legend.append('true ' + str(i))
        pyplot.plot( np.arange(len(speeds)),     speeds[:,i])
        legend.append('pred ' + str(i))

    pyplot.legend(legend)
    pyplot.show()        

    legend = []
    for i in range(len(mag_epochs)):
        mag_epochs[i] = transform_np(mag_epochs[i], [0,0,1,0])
        mag_epochs[i] = transform_np(mag_epochs[i], orient)
        mag_epochs[i] = transform_np(mag_epochs[i], inv_np(quats[i+1])) - north
    print(np.mean(mag_epochs[:,1]))

    for i in range(3):
        pyplot.plot( np.arange(len(mag_epochs)),     mag_epochs[:,i] + i/2)
        legend.append('src ' + str(i))
   
    '''
    for i in range(len(mag_epochs)):
        q = quats[i+1]
        q[3] = -q[3]
        mag_epochs[i] = transform_np(north, q)
        #mag_epochs[i] = transform_np(mag_epochs[i], fixed_trans)
    for i in range(3):
        pyplot.plot( np.arange(len(mag_epochs)),     mag_epochs[:,i])
        legend.append('nrd ' + str(i))
    '''

    pyplot.legend(legend)
    pyplot.show()        

    length = min(acs1.shape[0], acs_epochs.shape[0])
    acs1 = acs1[:length]
    acs_epochs = acs_epochs[:length]
    for i in range(len(acs_epochs) - 2):
        acs_epochs[i] = transform_np(acs_epochs[i], orient)
        acs_epochs[i] = transform_np(acs_epochs[i], quats[i+1])
        acs_epochs[i][2] -= g
        acs_epochs[i] = transform_np(acs_epochs[i], inv_np(quats[i+1]))
        acs1[i] = transform_np(acs1[i], inv_np(quats[i+1]))


    legend = []
    for i in range(3):
        pyplot.plot( np.arange(len(acs_epochs)),     acs_epochs[:,i] + i*5)
        legend.append('pred ' + str(i))
        pyplot.plot( np.arange(len(acs1)),     acs1[:,i] + i*5)
        legend.append('true ' + str(i))

    pyplot.legend(legend)
    pyplot.show()        

class RigidModel(tf.keras.layers.Layer):
    def __init__(self, num_epochs, **kwargs):
        super(RigidModel, self).__init__(**kwargs)

        self.num_epochs = num_epochs
        self.north = tf.Variable([1.,0.,0.], trainable=False, dtype = tf.float32)
        self.down  = tf.Variable([0.,0.,-1.], trainable=False, dtype = tf.float32)


    def build(self, input_shape):
        super(RigidModel, self).build(input_shape)

        self.g = self.add_weight(name='g', shape=(1, 1), trainable=True)
        self.speed = self.add_weight(name='speed', shape=(self.num_epochs+1,3), trainable=True)
        self.quat  = self.add_weight(name='orientation', shape=(self.num_epochs+1,4), trainable=True)

    def call(self, inputs):

        epoch_input, acs_input, gyr_input, mag_input = inputs

        ort, _ = tf.linalg.normalize(mag_input, axis = -1)

        indicies = tf.reshape(epoch_input,[-1])
        qt1 = tf.gather(self.quat, indicies, axis = 0)
        qt2 = tf.gather(self.quat, indicies - 1, axis = 0)
        qt1, _ = tf.linalg.normalize(qt1, axis = -1)
        qt2, _ = tf.linalg.normalize(qt2, axis = -1)

        speed1 = tf.gather(self.speed, indicies, axis = 0)
        speed2 = tf.gather(self.speed, indicies - 1, axis = 0)

        acsi = transform(acs_input,qt1)  - tf.reshape(self.down,[1,3])*tf.reshape(self.g,[1,1])
        acsr = speed1 - speed2

        mag_loss = tf.reduce_mean(tf.square(transform(ort,qt1) - tf.reshape(self.north,(1,3)))) / 10
        quat_loss = tf.reduce_mean(tf.square(mult(qt2,gyr_input)-qt1))
        acs_loss = tf.reduce_mean(tf.square(acsi-acsr)) *1e-5
        speed_loss = tf.reduce_mean(tf.square(speed1))*1e-8
        bnd_loss = tf.reduce_mean(tf.square(self.speed[1]) + tf.square(self.speed[-2]))*1e-2

        self.add_loss(mag_loss + quat_loss + acs_loss + bnd_loss)
        self.add_metric(mag_loss,'mag')
        self.add_metric(quat_loss,'quat')
        self.add_metric(acs_loss,'acs')
        self.add_metric(speed_loss,'speed')
        self.add_metric(bnd_loss,'bnd')
        self.add_metric(self.g,'g')

        return [qt1, speed1]

    def compute_output_shape(self, _):
        return (1)



def modelNhz(acsvalues,acstimes,gyrvalues,gyrtimes,magvalues,magtimes, hz, course):
    hz = 1
    beg_time = max(acstimes[0],gyrtimes[0],magtimes[0]) - 1000
    end_time = max(acstimes[-1],gyrtimes[-1],magtimes[-1])
    nummesures = int((end_time-beg_time)*hz/1000) + 1
    
    quat_epochs = np.zeros((nummesures,4))
    gyr_epochs = np.zeros((nummesures,3))
    acs_epochs = np.zeros((nummesures,3))
    mag_epochs = np.zeros((nummesures,3))
    mag_epochs[:,2] = 1
    quat_epochs[:,3] = 1

    for i in range(1, len(acstimes)):
        epoch = int((acstimes[i] - beg_time)*hz/1000 + 0.5)
        if epoch < 0 or epoch >= nummesures:
            continue
        dt = (acstimes[i] - acstimes[i-1])/1000
        acs_epochs[epoch] += acsvalues[i]*dt

    for i in range(1, len(magtimes)):
        epoch = int((magtimes[i] - beg_time)*hz/1000 + 0.5)
        if epoch < 0 or epoch >= nummesures:
            continue
        dt = (magtimes[i] - magtimes[i-1])/1000
        mag_epochs[epoch] += magvalues[i]*dt

    for i in range(1, len(gyrtimes)):
        epoch = int((gyrtimes[i] - beg_time)*hz/1000 + 0.5)
        if epoch < 0 or epoch >= nummesures:
            continue
        dt = (gyrtimes[i] - gyrtimes[i-1])/1000
        gyr_epochs[epoch] += gyrvalues[i]*dt

    for i in range(nummesures):
        quat_epochs[i] = to_quat(gyr_epochs[i], 1)

    quat_epochs2 = np.zeros((nummesures//hz + 1,4))
    quat_epochs2[:,3] = 1
    for i in range(nummesures):
        quat_epochs2[i//hz] = mult_np(quat_epochs[i], quat_epochs2[i//hz])
    quat_epochs2 = quat_epochs2/np.linalg.norm(quat_epochs2, axis = -1, keepdims=True)

    quat_epochs3 = np.zeros((len(course),4))
    quat_epochs3[:,3] = 1
    for i in range(1, len(course)):
        quat_epochs3[i] = get_quat(course[i], course[i-1])
    quat_epochs3 = quat_epochs3/np.linalg.norm(quat_epochs3, axis = -1, keepdims=True)


    # pyplot.plot( np.arange(len(quat_epochs2)),     quat_epochs2[:,0])
    # pyplot.plot( np.arange(len(quat_epochs2)),     quat_epochs2[:,1])
    # pyplot.plot( np.arange(len(quat_epochs2)),     quat_epochs2[:,2])
    # pyplot.plot( np.arange(len(quat_epochs2)),     quat_epochs2[:,3])
    # pyplot.plot( np.arange(len(quat_epochs3)),     quat_epochs3[:,0])
    # pyplot.plot( np.arange(len(quat_epochs3)),     quat_epochs3[:,1])
    # pyplot.plot( np.arange(len(quat_epochs3)),     quat_epochs3[:,2])
    # pyplot.plot( np.arange(len(quat_epochs3)),     quat_epochs3[:,3])
    # legend = ['px','py','pz','pw','tx','ty','tz','tw',]
    # pyplot.legend(legend)
    # pyplot.show()        

    rigid_model = RigidModel(nummesures+1)


    epoch_input = tf.keras.layers.Input((1,), dtype=tf.int32, name = 'epoch')
    acs_input = tf.keras.layers.Input((3,), dtype=tf.float32, name = 'acs')
    gyr_input = tf.keras.layers.Input((4,), dtype=tf.float32, name = 'gyr')
    mag_input = tf.keras.layers.Input((3,), dtype=tf.float32, name = 'mag')

    allinput = [epoch_input,acs_input,gyr_input,mag_input]
    alloutput = rigid_model(allinput)

    imu_model = tf.keras.Model(allinput, alloutput)
    imu_model.compile(tf.keras.optimizers.Adam(0.01))
    imu_model.summary()

    acs_epochs[np.isnan(acs_epochs)] = 0
    quat_epochs[np.isnan(quat_epochs)] = 0
    mag_epochs[np.isnan(mag_epochs)] = 0

    ds = tf.data.Dataset.from_tensor_slices({"epoch": np.arange(nummesures)+1, "acs": acs_epochs, "gyr": quat_epochs, "mag": mag_epochs})
    ds = ds.repeat()
    ds = ds.shuffle(4096)
    ds = ds.batch(1024)



    imu_model.fit(ds, epochs = 1024, steps_per_epoch = 1024)

from ..tf.tf_imu_model import createRigidModel
def analyze(logs, ground_truth, path):
    ground_times = ground_truth['times']
    ground_values = ground_truth['values']

    course = np.zeros(ground_values.shape)
    first_pos = 0
    mark_pos = 0
    last_pos  = 1
    while last_pos < len(ground_values):
        mv = ground_values[last_pos] - ground_values[first_pos]
        if np.linalg.norm(mv) > 0.5:
            mvd = mv/np.linalg.norm(mv)
            while mark_pos < (last_pos + first_pos)/2:
                course[mark_pos ]= mvd
                mark_pos += 1
            first_pos += 1
        else:
            last_pos += 1
    while mark_pos < last_pos:
        course[mark_pos ]= mvd
        mark_pos += 1

    rigid_model_layer = createRigidModel(ground_times*1000000,logs)
    optimizer = tf.keras.optimizers.Adam(0.1)
    quats = np.zeros((course.shape[0],4))
    quats[:,3] = 1
    fwd = np.array([0.,1.,0])
    for i in range(course.shape[0]):
        quats[i] = get_quat(fwd, course[i])

    #tf.keras.layer


    for epoch in range(1<<20):
        with tf.GradientTape() as tape:

            imu = rigid_model_layer((ground_values,quats), training=True)
            acs_loss = imu[0]
            quat_loss = imu[1]
            mag_loss = imu[2]
            speed_loss = imu[3]
            g = imu[4]
            total_loss = acs_loss+quat_loss+mag_loss+speed_loss

        print(epoch, acs_loss.numpy(), quat_loss.numpy(), mag_loss.numpy(), speed_loss.numpy(), g.numpy(), end = '\r')
        grads = tape.gradient(total_loss, rigid_model_layer.trainable_weights)
        optimizer.apply_gradients(zip(grads, rigid_model_layer.trainable_weights))        
        if epoch%64 == 0:
            print()


    acs = logs['acs']
    gir = logs['gir']
    mag = logs['mag']

    girtimes = gir['utcTimeMillis'].to_numpy()
    girvalues = gir[['UncalGyroXRadPerSec','UncalGyroYRadPerSec','UncalGyroZRadPerSec']].to_numpy()
    acstimes = acs['utcTimeMillis'].to_numpy()
    acsvalues = acs[['UncalAccelXMps2','UncalAccelYMps2','UncalAccelZMps2']].to_numpy()
    magtimes = mag['utcTimeMillis'].to_numpy()
    magvalues = mag[['UncalMagXMicroT','UncalMagYMicroT','UncalMagZMicroT']].to_numpy()

    
    rigidModelOnTruthData(acsvalues,acstimes,girvalues,girtimes,magvalues,magtimes, ground_values, course, ground_times, logs['baseline'], path)
    return
    #modelNhz(acsvalues,acstimes,girvalues,girtimes,magvalues,magtimes,100, course)


    local_trans = np.zeros((ground_values.shape[0],3,3))
    local_trans[:,2,:] = [0,0,1]
    local_trans[:,1,:] = course
    local_trans[:,0,:] = np.cross(local_trans[:,1], local_trans[:,2])
    local_trans = np.transpose(local_trans, (0,2,1))


    acsnorms = np.linalg.norm(acsvalues,axis=-1) 
    acsnorms = np.sort(acsnorms)
    acsvalue = np.median(acsnorms)
    print(acsvalue)
    acsmean = np.mean(acsvalues,axis=0,keepdims=True)
    acsvalues -= acsmean
    print(acsmean)
    print(np.linalg.norm(acsmean))
    
    window = 15000
    acsmedian = np.stack([running_median_insort(acsvalues[:,0], window), running_median_insort(acsvalues[:,1], window), running_median_insort(acsvalues[:,2], window)], axis = 1)
    acsvalues -= acsmedian

    speed = ground_values[1:] - ground_values[:-1]
    accel = speed[1:] - speed[:-1]
    accel = np.reshape(accel,(-1,1,3))
    accellocal = np.matmul(accel, local_trans[1:-1])
    accellocal = np.reshape(accellocal,(-1,3))



    girmedian = np.stack([running_median_insort(girvalues[:,0], window), running_median_insort(girvalues[:,1], window), running_median_insort(girvalues[:,2], window)], axis = 1)
    girvalues -= girmedian




    showIntegrate(acsvalues, acstimes, accellocal)
    showIntegrate(acsmedian, acstimes, accellocal)

    return

    ground_times = ground_truth['times']
    ground_values = ground_truth['values']
    

