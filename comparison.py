import pandas as pd
import os
import numpy as np
import re
from matplotlib import pyplot as plt
# print(os.listdir("./data"))

def test_errors():
    frequencies_tested = sorted([int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir("./data/") if "quaternions" in file])

    reference_freq = np.max(frequencies_tested)


    reference_data = pd.read_csv("./data/quaternions_freq{}.csv".format(reference_freq))
    reference_pos = pd.read_csv("./data/positions_freq{}.csv".format(reference_freq))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for freq in frequencies_tested:
        if freq != reference_freq:
            freq_data = pd.read_csv("./data/quaternions_freq{}.csv".format(freq))
            pos_data = pd.read_csv("./data/positions_freq{}.csv".format(freq))
            squared_error_freq = (freq_data-reference_data)**2
            squared_error_pos = (pos_data-reference_pos)**2
            RMS_freq = np.log(np.sqrt(np.sum(squared_error_freq[squared_error_freq.columns[1:]], axis=1)))
            RMS_pos = np.sqrt(np.sum(squared_error_pos[squared_error_pos.columns[1:]], axis=1))
            # print(np.log(RMS))
            ax1.plot(range(len(freq_data)), RMS_freq, label="freq={}".format(freq))
            ax2.plot(range(len(pos_data)), RMS_pos, label="freq={}".format(freq))

    # ax1.set_xlabel("time step")
    ax1.set_ylabel("log of RMS error")
    ax1.set_title("log of RMS of quaternion components")
    # ax1.title("Error is defined as\nsquare root of sum of squared difference of components")
    ax2.set_xlabel("time step")
    ax2.set_title("Error in Position")
    ax2.set_ylabel("RMS of position")
    plt.legend(loc='best')
    plt.show()


frequencies_tested = np.unique(sorted([int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir("./data/") if "quat_" in file]))
thresholds_tested = np.unique(sorted([float(re.findall(r'\d+(\.\d+)?', file)[1]) for file in os.listdir("./data/") if "quat_" in file]))
n,m = len(frequencies_tested), len(thresholds_tested)
print(frequencies_tested)
print(thresholds_tested)

ref_freq = 256
ref_tau = 0.05

ref_data_freq = pd.read_csv("./data/quat_freq256_thresh0.05.csv")
ref_data_pos = pd.read_csv("./data/pos_freq256_thresh0.05.csv")

final_pos_error = np.zeros((n,m))
print(n,m)
print(final_pos_error)
for freq in frequencies_tested:
    for tau in thresholds_tested:
        freq_data = pd.read_csv("./data/quat_freq{}_thresh{}.csv".format(freq, tau))
        pos_data = pd.read_csv("./data/pos_freq{}_thresh{}.csv".format(freq, tau))
        squared_error_freq = (ref_data_freq-freq_data)**2
        squared_error_pos = (ref_data_pos-pos_data)**2

        rms_freq = np.sqrt(np.sum(squared_error_freq[squared_error_freq.columns[1:]], axis=1))
        rms_pos = np.sqrt(np.sum(squared_error_pos[squared_error_pos.columns[1:]], axis=1))
        # print(rms_pos.iloc[-1])
        # quit()
        i = np.where(frequencies_tested==freq)
        j = np.where(thresholds_tested==tau)
        final_pos_error[i,j] = rms_pos.iloc[-1]
        # plt.plot(range(len(rms_pos)), rms_pos, label=r"freq={}, \tau={}".format(freq, tau))
# plt.legend(loc='best')
fig, ax = plt.subplots()
plt.imshow(final_pos_error)
plt.xlabel("threshold")
# plt.xticks(thresholds_tested)
ax.set_xticks(thresholds_tested)
plt.ylabel("frequency tested")
plt.colorbar()
plt.show()

