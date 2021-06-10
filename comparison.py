import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

frequencies_tested = sorted([int(file[16:19]) for file in os.listdir() if "quaternions" in file])


reference_freq = np.max(frequencies_tested)

reference_data = pd.read_csv("quaternions_freq{}.csv".format(reference_freq))
reference_pos = pd.read_csv("positions_freq{}.csv".format(reference_freq))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for freq in frequencies_tested:
    if freq != reference_freq:
        freq_data = pd.read_csv("quaternions_freq{}.csv".format(freq))
        pos_data = pd.read_csv("positions_freq{}.csv".format(freq))
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

# test positions
