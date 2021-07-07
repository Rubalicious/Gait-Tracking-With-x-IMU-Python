import ahrs
from ahrs import Quaternion
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import argparse
from ahrs import QuaternionArray
from skimage import restoration, filters
import pylops
import pandas as pd

# option = 'IMU' # or 'MARG'
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-a', '--algorithm',
#     default='IMU',
#     help='choose between IMU and MARG')
# parser.add_argument(
#     '-f', '--frequency',
#     default='256',
#     help='choose sampling frequency to use'
# )
# parser.add_argument(
#     '-s', '--scaling',
#     default=1,
#     help="factor by which to subsample data"
# )
# parser.add_argument(
#     '-t', '--tau',
#     default=0.1,
#     help="specify the acceleration threshold"
# )
# args = parser.parse_args()
# option = args.algorithm
# sf = float(args.frequency)
# tau = float(args.tau)
# scaling = float(args.scaling)


# filePath = 'datasets/straightLine'
# startTime = 6
# stopTime = 26
# samplePeriod = 1/256

# filePath = 'datasets/stairsAndCorridor'
# startTime = 5
# stopTime = 53
# samplePeriod = 1/256

# filePath = 'datasets/spiralStairs'
# startTime = 4
# stopTime = 47
# samplePeriod = 1/256

filePath = './example_Data'
startTime=0
stopTime=100
samplePeriod=1/256

def build_trajectory(freq=256, tau=0.05, alg="Madgwick", plot_graphs=False, use_MARG=False):
    # -------------------------------------------------------------------------
    # extract data
    xIMUdata = xIMU.xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)
    time = xIMUdata.CalInertialAndMagneticData.Time
    gyrX = xIMUdata.CalInertialAndMagneticData.gyroscope[:,0]
    gyrY = xIMUdata.CalInertialAndMagneticData.gyroscope[:,1]
    gyrZ = xIMUdata.CalInertialAndMagneticData.gyroscope[:,2]
    accX = xIMUdata.CalInertialAndMagneticData.accelerometer[:,0]
    accY = xIMUdata.CalInertialAndMagneticData.accelerometer[:,1]
    accZ = xIMUdata.CalInertialAndMagneticData.accelerometer[:,2]
    # print(xIMUdata.CalInertialAndMagneticData.magnetometer)
    # quit()
    magX = xIMUdata.CalInertialAndMagneticData.magnetometer[:,0]
    magY = xIMUdata.CalInertialAndMagneticData.magnetometer[:,1]
    magZ = xIMUdata.CalInertialAndMagneticData.magnetometer[:,2]

    indexSel = np.all([time>=startTime,time<=stopTime], axis=0)
    time = time[indexSel]
    gyrX = gyrX[indexSel]
    gyrY = gyrY[indexSel]
    gyrZ = gyrZ[indexSel]
    accX = accX[indexSel]
    accY = accY[indexSel]
    accZ = accZ[indexSel]
    magX = magX[indexSel]
    magY = magY[indexSel]
    magZ = magZ[indexSel]

    # numSamples = len(time)

    # factor = scaling # factor by which to subsample data

    T = len(time)*samplePeriod # total time
    # N = int((T/samplePeriod)/factor)
    # sample_frequency = N/T
    sample_frequency=float(freq)
    N = int(sample_frequency*T)
    # print("test new sample frequency is {} Hz".format(sample_frequency))


    # print("length of time series is {}".format(N))
    # print("threshold = {}".format(float(tau)))

    # -------------------------------------------------------------------------
    # subsampling accX_ss = accX_subsampled
    time = signal.resample(time, N)

    accX = signal.resample(accX, N)
    accY = signal.resample(accY, N)
    accZ = signal.resample(accZ, N)

    gyrX = signal.resample(gyrX, N)
    gyrY = signal.resample(gyrY, N)
    gyrZ = signal.resample(gyrZ, N)

    magX = signal.resample(magX, N)
    magY = signal.resample(magY, N)
    magZ = signal.resample(magZ, N)

    # -------------------------------------------------------------------------
    # signal processing
    # take a derivative, filter out high frequency data (not corresponding to walking ) 5-6 Hz
    # butterworth

    # Compute accelerometer magnitude
    acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)

    # Compute magentometer magnitude
    mag_mag = np.sqrt(magX*magX+magY*magY+magZ*magZ)

    # Compute gyroscope magnitude
    gyr_mag = np.sqrt(gyrX*gyrX+gyrY*gyrY+gyrZ*gyrZ)

    # HP filter accelerometer data
    # Hodrick-Prescott filter: removes cyclical components of signal
    filtCutOff = 0.001
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'highpass')
    acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    # Compute absolute value
    acc_magFilt = np.abs(acc_magFilt)

    # LP filter accelerometer data
    # Low-Pass filter: use butterworth filter to filter out high frequency data
    filtCutOff = 5
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass') #'lowpass'
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))


    # Ruby's contribution
    filtCutOff_mag = 2
    bm, am = signal.butter(1, (2*filtCutOff_mag)/(1/samplePeriod), 'lowpass') #'lowpass'
    mag_magFilt = signal.filtfilt(bm, am, mag_mag, padtype = 'odd', padlen=3*(max(len(bm),len(am))-1))

    b,a = signal.butter(1, (2*filtCutOff_mag)/(1/samplePeriod), 'lowpass')
    magX_Filt = signal.filtfilt(b, a, magX, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    mag_der = np.diff(magX, n=1)

    filtCutOff_mag = 10
    bm, am = signal.butter(1, (2*filtCutOff_mag)/(1/samplePeriod), 'lowpass') #'lowpass'
    mag_derFilt = signal.filtfilt(bm, am, mag_der, padtype = 'odd', padlen=3*(max(len(bm),len(am))-1))

    der_threshold = .2*(np.abs(mag_derFilt) > .005)



    # -------------------------------------------------------------------------
    # denoising magentic field strength data using total variation
    # magX = restoration.denoise_tv_bregman(magX, weight=.7)
    # magX = restoration.denoise_tv_chambolle(magX, weight=.5)
    Iop = pylops.Identity(N)
    y = Iop*magX

    Dop = pylops.FirstDerivative(N, edge=True, kind='backward')
    mu = 0.1     #0.01
    lamda = 1   #0.3
    niter_out = 50 #50
    niter_in = 3

    xinv, niter = \
        pylops.optimization.sparsity.SplitBregman(Iop, [Dop], y, niter_out,
                                                  niter_in, mu=mu, epsRL1s=[lamda],
                                                  tol=1e-4, tau=1.,
                                                  **dict(iter_lim=30, damp=1e-10))


    # Threshold detection
    # tau = 0.1
    stationary = acc_magFilt < tau # originally set to 0.05

    stationary_gyr = gyr_mag < 40




    if plot_graphs: 
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3) # 3,1,3
        ax1.plot(time,gyrX,c='r',linewidth=0.5)
        ax1.plot(time,gyrY,c='g',linewidth=0.5)
        ax1.plot(time,gyrZ,c='b',linewidth=0.5)
        ax1.plot(time,gyr_mag, c='k',linestyle=":")
        ax1.plot(time,100*stationary_gyr, c='k', linewidth=2)

        ax1.set_title("gyroscope")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("angular velocity (degrees/s)")
        ax1.legend(["x","y","z"])
        ax2.plot(time,accX,c='r',linewidth=0.5)
        ax2.plot(time,accY,c='g',linewidth=0.5)
        ax2.plot(time,accZ,c='b',linewidth=0.5)
        ax2.plot(time,acc_magFilt,c='k',linestyle=":",linewidth=1)
        ax2.plot(time,stationary,c='k')
        ax2.set_title("accelerometer")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("acceleration (g)")
        ax2.legend(["x","y","z"])

        ax3.plot(time, magX, c='r', linewidth=0.5)
        ax3.plot(time, magY,c='g',linewidth=0.5)
        ax3.plot(time, magZ,c='b',linewidth=0.5)
        norm = np.sqrt(magX*magX+magY*magY)
        ax3.plot(time, norm,c='k',linewidth=0.5)
        # ax3.plot(time, mag_mag,c='k', linewidth=0.5, linestyle=':')
        # ax3.plot(time[:-1], mag_derFilt,c='k', linewidth=0.5, linestyle='--')
        # ax3.plot(time[:-1], der_threshold,c='k', linewidth=1, linestyle='-')
        # ax3.plot(time, xinv, linewidth=1, label="pylops")

        ax3.set_title("magnetometer with {} samples".format(N))
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("magnetic field strength (mT)")
        ax3.legend(["x","y","z"])
        ax3.legend()
        plt.show(block=False)
        # plt.show()

    # Compute orientation
    quat  = np.zeros((time.size, 4), dtype=np.float64)

    # initial convergence
    initPeriod = 2
    indexSel = time<=time[0]+initPeriod
    gyr = np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
    mag = np.array([np.mean(magX[indexSel]), np.mean(magY[indexSel]), np.mean(magZ[indexSel])])

    # types of filters
    mahony = ahrs.filters.Mahony(Kp=1, Ki=0, KpInit=1, Dt=1/sample_frequency)
    madgwick = ahrs.filters.Madgwick(Dt=1/sample_frequency)
    aqua = ahrs.filters.AQUA(Dt=1/sample_frequency)
    comp = ahrs.filters.Complementary(frequency=1/samplePeriod)
    ekf = ahrs.filters.EKF(Dt=1/sample_frequency)
    fourati = ahrs.filters.Fourati(Dt=1/sample_frequency)

    # initial convergence
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    for i in range(2000):
        if alg == 'Madgwick':
            if use_MARG:
                q = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            else:
                q = madgwick.updateIMU(q, gyr=gyr, acc=acc) #, mag=mag  # updateIMU # updateMARG # update
        elif alg == 'Mahony':
            if use_MARG:
                q = mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            else:
                q = mahony.updateIMU(q, gyr=gyr, acc=acc)
        elif alg == 'Fourati':
            if not use_MARG: mag = []
            q = fourati.update(q, gyr=gyr, acc=acc, mag=mag)

    # all data can be returned in this form
    # orientation = Fourati(gyr)
    # orientation = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data)
    # orientation = Mahony(gyr=gyro_data, acc=acc_data, mag=mag_data) 

    # For all data
    for t in range(0,time.size):
        if(stationary[t]):
            mahony.Kp = 0.5 # 0.5
        else:
            mahony.Kp = 0.0 # 0
        gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        acc = np.array([accX[t],accY[t],accZ[t]])
        mag = np.array([magX[t],magY[t],magZ[t]])
        # if option == 'IMU':
        # quat[t,:]=madgwick.updateIMU(q,gyr=gyr,acc=acc) # , mag=mag # updateIMU # updateMARG # update
        # elif option == 'MARG':
            # quat[t,:]=ekf.update(q,gyr=gyr,acc=acc, mag=mag)
        if alg == 'Madgwick':
            if use_MARG:
                quat[t,:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            else:
                quat[t,:] = madgwick.updateIMU(q, gyr=gyr, acc=acc) #, mag=mag  # updateIMU # updateMARG # update
        elif alg == 'Mahony':
            if use_MARG:
                quat[t,:] = mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            else:
                quat[t,:] = mahony.updateIMU(q, gyr=gyr, acc=acc)
            
        elif alg == 'Fourati':
            if not use_MARG: mag = []
            quat[t,:] = fourati.update(q, gyr=gyr, acc=acc, mag=mag)


    # -------------------------------------------------------------------------
    # Compute translational accelerations

    # Rotate body accelerations to Earth frame
    acc = []
    for x,y,z,q in zip(accX,accY,accZ,quat):
        acc.append(q_rot(q_conj(q), np.array([ x, y, z])))
    acc = np.array(acc)
    acc = acc - np.array([0,0,1])
    acc = acc * 9.81

    # Compute translational velocities
    # acc[:,2] = acc[:,2] - 9.81

    # acc_offset = np.zeros(3)
    vel = np.zeros(acc.shape)
    for t in range(1,vel.shape[0]):
        vel[t,:] = vel[t-1,:] + acc[t,:]/sample_frequency #*samplePeriod
        if stationary[t] == True:
            vel[t,:] = np.zeros(3)

    # Compute integral drift during non-stationary periods
    velDrift = np.zeros(vel.shape)
    stationaryStart = np.where(np.diff(stationary.astype(int)) == -1)[0]+1  
    stationaryEnd = np.where(np.diff(stationary.astype(int)) == 1)[0]+1
    # print(stationaryStart, stationaryEnd)
    # quit()
    for i in range(0,stationaryEnd.shape[0]):
        driftRate = vel[stationaryEnd[i]-1,:] / (stationaryEnd[i] - stationaryStart[i])
        enum = np.arange(0,stationaryEnd[i]-stationaryStart[i])
        drift = np.array([enum*driftRate[0], enum*driftRate[1], enum*driftRate[2]]).T
        velDrift[stationaryStart[i]:stationaryEnd[i],:] = drift

    # Remove integral drift
    vel = vel - velDrift
    if plot_graphs: 
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(211)
        ax1.plot(time,vel[:,0],c='r',linewidth=0.5)
        ax1.plot(time,vel[:,1],c='g',linewidth=0.5)
        ax1.plot(time,vel[:,2],c='b',linewidth=0.5)
        ax1.legend(["x","y","z"])
        ax1.set_title("velocity sampled at {:.2f} Hz".format(sample_frequency))
        # ax1.set_xlabel("time (s)")
        ax1.set_ylabel("velocity (m/s)")
        plt.show(block=False)

    # -------------------------------------------------------------------------
    # Compute translational position
    pos = np.zeros(vel.shape)
    for t in range(1,pos.shape[0]):
        pos[t,:] = pos[t-1,:] + vel[t,:]/sample_frequency #*samplePeriod

    # fig = plt.figure(figsize=(10, 5))
    if plot_graphs: 
        ax2 = fig.add_subplot(212)
        ax2.plot(time,pos[:,0],c='r',linewidth=0.5)
        ax2.plot(time,pos[:,1],c='g',linewidth=0.5)
        ax2.plot(time,pos[:,2],c='b',linewidth=0.5)
        ax2.legend(["x","y","z"])
        ax2.set_title("position sampled at {:.2f} Hz".format(sample_frequency))
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("position (m)")
        plt.show(block=False)

    # -------------------------------------------------------------------------
    # Plot 3D foot trajectory

    posPlot = pos
    quatPlot = quat

    extraTime = 20
    # onesVector = np.ones(int(extraTime*(1/samplePeriod)))

    # Create 6 DOF animation
    if plot_graphs: 
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d') # Axe3D object


        ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
        # for i in range(0, len(posPlot), 200):
        #     A = Quaternion(quat[i]).to_DCM()
        #     ax.quiver(posPlot[i,0],posPlot[i,1],posPlot[i,2], A[:, 0], A[:,1], A[:, 2], color=['r','g','b']) # modify this to quiver.
        # print(quatPlot[-1])
        min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
        ax.set_xlim(min_,max_)
        ax.set_ylim(min_,max_)
        ax.set_zlim(min_,max_)
        ax.set_title('trajectory using {}\nand sample frequency {:.2f}\nthreshold used: {:.2f}'.format(alg, sample_frequency, tau))
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_zlabel("z position (m)")

        plt.show(block=False)
        plt.show()

    # -------------------------------------------------------------------------
    # upsample and record data
    # quat = signal.resample(quat, numSamples)
    # data = pd.DataFrame(quat)
    # data.to_csv("./data/quat_freq{}_thresh{}.csv".format(int(sample_frequency), tau))
    # pos = signal.resample(pos, numSamples)
    # data = pd.DataFrame(pos)
    # data.to_csv("./data/pos_freq{}_thresh{}.csv".format(int(sample_frequency), tau))

    return pos, quat, vel


if __name__ == "__main__":
    pos, quat, vel = build_trajectory(freq=256, tau=0.05, plot_graphs=True, alg="Madgwick") # reference data
    # factors = np.linspace(1.0, 3.0, 10)
    # taus = np.linspace(0.05, 0.1, 10)
    # print(factors, taus)

    arclength = 0
    for vx,vy,vz in vel:
        arclength += np.sqrt(vx**2+vy**2+vz**2)
    print("arclength of trajectory = {}".format(arclength))

