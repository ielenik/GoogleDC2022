
from numpy import linalg
from src.laika.lib.coordinates import ecef2geodetic, geodetic2ecef
from src.laika import AstroDog
from src.laika.gps_time import GPSTime

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.training.tracking import base

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from matplotlib import pyplot
from .tf_phone_model import createGpsPhoneModel
from .tf_inertial_model import create_inertial_model
from .tf_track_model import createTrackModel
from src.utils.kml_writer import KMLWriter
from collections import deque
from bisect import insort, bisect_left
from itertools import islice
from . import tf_rigid_model
from . import tf_imu_model
from .tf_numpy_tools import transform

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
autotune = tf.data.experimental.AUTOTUNE
tf.keras.backend.set_floatx('float32')

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

def get_track(measures, folder, wgs_converter):
    if True:
        trajectory = []
        true_values = False
        for m in measures:
            #if False:
            if 'groundtrue' in m:
                trajectory.extend(zip(m['groundtrue']['times']*1000000,m['groundtrue']['values']))
                true_values = True
            else:
                trajectory.extend(zip(m['epoch_times'],m['baseline']))

        trajectory = sorted(trajectory)

        times_baseline = times = [item[0] for item in trajectory ]
        coords_baseline = coords = [item[1] for item in trajectory ]
        coords = np.array(coords)

        if not true_values:
            window = 16
            coords[:,0] = running_median_insort(coords[:,0], window)
            coords[:,1] = running_median_insort(coords[:,1], window)
            coords[:,2] = running_median_insort(coords[:,2], window)
        
        for m in measures:
            tri = 0
            for i in range(len(m['epoch_times'])):
                while tri < len(times) and times[tri] < m['epoch_times'][i]:
                    tri+=1
                if tri == len(times):
                    tri -= 1
                m['baseline'][i] = coords[tri]

    phone_models = []
    imu_models = []
    phone_times  = []

    max_time = min_time = 0

    for m in measures:
        m['sat_deltavalid'][np.isnan(m['sat_deltarange'])] = 0
        m['sat_deltarange'][np.isnan(m['sat_deltarange'])] = 0
        m['sat_positions'][np.isnan(m['sat_positions'])] = 1
        model, times = createGpsPhoneModel(m)
        phone_models.append(model)
        phone_times.append(times)
        imu_models.append(tf_imu_model.createRigidModel(times, m))

        if min_time == 0 or min_time > times[0]:
            min_time = times[0]
        if max_time == 0 or max_time < times[-1]:
            max_time = times[-1]
        
    model_track, track_model_error, num_measures, start_nanos, time_tick = tf_rigid_model.createTrackModel(min_time,max_time, {'times':times_baseline,'values':coords_baseline} )

    track_input = np.arange(num_measures)
    track_input = np.reshape(track_input,(-1,1))

    forward = tf.Variable([[0.,1.,0.]], trainable=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001)#, epsilon= 0.0001)
    if False:
        @tf.function
        def train_step_gnss(optimizer):
            @tf.custom_gradient
            def norm(x):
                y = tf.linalg.norm(x, 'euclidean', -1)
                def grad(dy):
                    return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

                return y, grad
            for _ in range(16):
                with tf.GradientTape(persistent=True) as tape:
                    total_loss_psevdo = 0
                    total_loss_delta = 0

                    accs_loss = 0
                    speed_loss_small = 0
                    for i in range(len(phone_models)):
                        poses, quats = model_track(phone_times[i], training=True)
                        poses = tf.reshape(poses,(1,-1,3))
                        psevdo_loss,delta_loss = phone_models[i](poses, training=True)
                        total_loss_psevdo += psevdo_loss/10
                        total_loss_delta  += delta_loss*2
                        
                    

                    poses, quats = track_model_error(track_input, training=True)
                    accs_loss = tf.reduce_mean(norm(2*poses[1:-1] - poses[2:] - poses[:-2]))
                    speed_loss_small = tf.reduce_mean(norm(poses[1:] - poses[:-1]))*0.001
                    


                    qt1 = (quats[1:]+quats[:-1])/2
                    dqt = (quats[1:]-quats[:-1])
                    speed = poses[1:] - poses[:-1]
                    #orient_loss = tf.reduce_mean((norm(speed) - tf.reduce_sum(speed*transform(forward,qt1), axis = -1))/(norm(speed) + 0.01)) + tf.reduce_mean(tf.square(dqt))
                    orient_loss = tf.reduce_mean((norm(speed) - tf.reduce_sum(speed*transform(forward,qt1), axis = -1)))/10 + tf.reduce_mean(tf.square(dqt))

                    total_loss = (accs_loss + speed_loss_small)*0.1 + total_loss_delta +total_loss_psevdo + orient_loss

                
                for i in range(len(phone_models)):
                    grads = tape.gradient(total_loss, phone_models[i].trainable_weights)
                    optimizer.apply_gradients(zip(grads, phone_models[i].trainable_weights))        


                grads = tape.gradient(total_loss, model_track.trainable_weights)
                optimizer.apply_gradients(zip(grads, model_track.trainable_weights))        
                grads = tape.gradient(total_loss, track_model_error.trainable_weights)
                optimizer.apply_gradients(zip(grads, track_model_error.trainable_weights))        
                
                del tape

            return  total_loss, accs_loss, speed_loss_small, total_loss_psevdo, total_loss_delta, orient_loss    
        
        lr = 0.01

        for step in range(32*10+1):
            for i in range(32):
                total_loss, accs_loss,  speed_loss_small, total_loss_psevdo, total_loss_delta, orient_loss = train_step_gnss(optimizer)

            print( "Training loss at step %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f  lr %.4f" % (step, 
                float(total_loss), 
                float(total_loss_psevdo), 
                float(total_loss_delta), 
                float(accs_loss), 
                float(speed_loss_small), 
                float(orient_loss), 
                float(lr)), end='\r')
            if(step % 32 == 0):
                lr *= 0.7
                optimizer.learning_rate = lr

                poses, quats = track_model_error(track_input)
                wgs_poses = wgs_converter(poses)
                kml = KMLWriter(folder+ "/predicted_"+str(step)+".kml", "predicted")
                kml.addTrack('predicted_'+str(step),'FF00FF00', wgs_poses)
                kml.finish()

                print()
            
            '''
            shift = pred_pos - gt_ecef_coords
            meanshift = np.mean(shift,axis=0,keepdims=True)
            shift = shift - meanshift
            err3d = np.mean(np.linalg.norm(shift,axis = -1))
            dist_2d = np.linalg.norm(shift[:,:2],axis = -1)
            err2d = np.mean(dist_2d)
            dist_2d = np.sort(dist_2d)
            err50 = dist_2d[len(dist_2d)//2]
            err95 = dist_2d[len(dist_2d)*95//100]

            delta_dif = delta_dif.numpy()
            delta_dif = delta_dif[np.abs(delta_dif) > 0]
            percents_good = np.sum(np.abs(delta_dif) < 0.1)*100/len(delta_dif)


            print( "Training loss at step %d (%.2f (%.2f),%.2f,%.2f,%.2f,%.4f): %.4f (%.2f),%.4f (%.2f),%.4f,%.4f,%.4f,%.4f  lr %.4f" % (step, err3d, np.linalg.norm(meanshift[0,:2]), err2d, err50, err95, (err50+err95)/2, float(total_loss_psevdo), percents_good_psev, float(total_loss_delta),percents_good,float(accs_loss_large),float(accs_loss_small), float(speed_loss_small), float(imu_loss), float(lr)), end='\r')
            if(step % 32 == 0):
                lr *= 0.7
                optimizer.learning_rate = lr
                print()
                if True:
                    plt.clf()
                    plt.scatter(pred_pos[:,1], pred_pos[:,0], s=0.2, alpha=0.5)
                    plt.scatter(gt_ecef_coords[:,1], gt_ecef_coords[:,0], s=0.2, alpha=0.5)
                    #fig1.canvas.start_event_loop(sys.float_info.min) #workaround for Exception in Tkinter callback
                    plt.savefig("fig/"+track+str(step+10000)+".svg", dF = 1000)
                    plt.close()
        index = 0
        for m in measures:
            phone_models[index].save(folder+'/phone_'+str(index)+'.h5')
            imu_models[index].save(folder+'/imu_'+str(index)+'.h5')
            np.save(folder+'/time_'+str(index)+'.npy',phone_times[index])
            index += 1
            '''
    
        
    @tf.function
    def train_step_gnss_with_accel(optimizer, gyro_weight):
        @tf.custom_gradient
        def norm(x):
            y = tf.linalg.norm(x, 'euclidean', -1)
            def grad(dy):
                return tf.expand_dims(dy,-1) * (x / (tf.expand_dims(y + 1e-19,-1)))

            return y, grad
        for _ in range(4):
            with tf.GradientTape(persistent=True) as tape:
                total_loss_psevdo = 0
                total_loss_delta = 0
                acs_loss =  quat_loss =  mag_loss = speed_loss = 0

                accs_loss = 0
                speed_loss_small = 0
                for i in range(len(phone_models)):
                    poses, quats = model_track(phone_times[i], training=True)
                    poses = tf.reshape(poses,(1,-1,3))
                    psevdo_loss,delta_loss = phone_models[i](poses, training=True)
                    total_loss_psevdo += psevdo_loss/10
                    total_loss_delta  += delta_loss*2
                    imu = imu_models[i]((poses,quats), training=True)
                    acs_loss += imu[0]
                    quat_loss += imu[1]
                    mag_loss += imu[2]
                    speed_loss += imu[3]
                    g = imu[4]
                    
                total_loss = total_loss_psevdo + total_loss_delta+(acs_loss*0.1 + mag_loss + quat_loss*gyro_weight  + speed_loss*10)#*0.1

            for i in range(len(phone_models)):
                grads = tape.gradient(total_loss, phone_models[i].trainable_weights)
                optimizer.apply_gradients(zip(grads, phone_models[i].trainable_weights))        
                grads = tape.gradient(total_loss, imu_models[i].trainable_weights)
                optimizer.apply_gradients(zip(grads, imu_models[i].trainable_weights))        


            grads = tape.gradient(total_loss, model_track.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_track.trainable_weights))        
            del tape

        return  total_loss, total_loss_psevdo, total_loss_delta, acs_loss, quat_loss, mag_loss,  speed_loss, g


    lr = 0.01
    #optimizer = tf.keras.optimizers.SGD(learning_rate=100., nesterov=True, momentum=0.5)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#, epsilon= 0.0001)
    #optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=100. )
    for step in range(769):
        for i in range(128):
            total_loss, total_loss_psevdo, total_loss_delta, acs_loss, quat_loss, mag_loss,  speed_loss, g  = train_step_gnss_with_accel(optimizer, tf.constant(1.))

        print( "Training loss at step %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.2f  lr %.4f" % (step, 
            float(total_loss), 
            float(total_loss_psevdo), 
            float(total_loss_delta), 
            float(acs_loss), 
            float(quat_loss), 
            float(mag_loss), 
            float(speed_loss), 
            float(g), 
            float(lr)), end='\r')
        if(step % 32 == 0):
            lr *= 0.9
            if lr < 0.0001:
                lr = 0.0001

            poses, quats = track_model_error(track_input)
            wgs_poses = wgs_converter(poses)
            kml = KMLWriter(folder+ "/predicted2_"+str(step)+".kml", "predicted")
            kml.addTrack('predicted2_'+str(step),'FF00FF00', wgs_poses)
            kml.finish()

            optimizer.learning_rate = lr
            print()

            '''
            pred_pos = model_track(baseline_times*1000000).numpy()
            poses = poses.numpy()
            psev_error = psev_error.numpy()
            psev_error = psev_error[np.abs(psev_error) > 0]
            percents_good_psev = np.sum(np.abs(psev_error) < 1)*100/len(psev_error)
            

            shift = pred_pos - gt_ecef_coords
            meanshift = np.mean(shift,axis=0,keepdims=True)
            shift = shift - meanshift
            err3d = np.mean(np.linalg.norm(shift,axis = -1))
            dist_2d = np.linalg.norm(shift[:,:2],axis = -1)
            err2d = np.mean(dist_2d)
            dist_2d = np.sort(dist_2d)
            err50 = dist_2d[len(dist_2d)//2]
            err95 = dist_2d[len(dist_2d)*95//100]

            delta_dif = delta_dif.numpy()
            delta_dif = delta_dif[np.abs(delta_dif) > 0]
            percents_good = np.sum(np.abs(delta_dif) < 0.1)*100/len(delta_dif)


            print( "Training loss at step %d (%.2f (%.2f),%.2f,%.2f,%.2f,%.4f): %.4f (%.2f),%.4f (%.2f),%.4f,%.4f,%.4f,%.4f  lr %.4f" % (step, err3d, np.linalg.norm(meanshift[0,:2]), err2d, err50, err95, (err50+err95)/2, float(total_loss_psevdo), percents_good_psev, float(total_loss_delta),percents_good,float(accs_loss_large),float(accs_loss_small), float(speed_loss_small), float(imu_loss), float(lr)), end='\r')
            if(step % 128 == 0):
                lr *= 0.9
                if lr < 0.001:
                    lr = 0.001

                optimizer.learning_rate = lr
                print()
                if True:
                    plt.clf()
                    plt.scatter(pred_pos[:,1], pred_pos[:,0], s=0.2, alpha=0.5)
                    plt.scatter(gt_ecef_coords[:,1], gt_ecef_coords[:,0], s=0.2, alpha=0.5)
                    #fig1.canvas.start_event_loop(sys.float_info.min) #workaround for Exception in Tkinter callback
                    plt.savefig("fig/"+track+str(step+10000)+".svg", dpi = 1000)
                    plt.close()
                if True:
                    true_accs = true_accs.numpy()
                    plt.clf()
                    plottimes = np.arange(len(true_accs)-1)
                    lgd = ['true_x', 'true_y']

                    colors = [ '#FF0000', '#00FF00', '#0000FF', '#00FFFF' ]
                    plt.plot( plottimes, true_accs[1:,0], '#000000')
                    plt.plot( plottimes, true_accs[1:,1]+ 3, '#000000')
                    #plt.plot( plottimes, true_accs[1:,2])

                    count = 0
                    for pred_acs in pred_acses:
                        pred_acs = pred_acs.numpy()
                        plt.plot( plottimes, pred_acs[1:,0], colors[count], alpha=0.3, linewidth=0.5)
                        plt.plot( plottimes, pred_acs[1:,1]+ 3, colors[count], alpha=0.3, linewidth=0.5)
                        lgd.append(phone_names[count] + ' x')
                        lgd.append(phone_names[count] + ' y')
                        count += 1
                    #plt.plot( plottimes, pred_acs[:,2])
                    plt.legend(lgd)
                    #plt.show()
                    plt.savefig("fig/acs"+track+str(step+10000)+".svg", dpi = 1000)
                    plt.close()
            
            '''

    poses, qr = track_model_error(track_input)
    times = start_nanos + time_tick*track_input
    return times, poses
