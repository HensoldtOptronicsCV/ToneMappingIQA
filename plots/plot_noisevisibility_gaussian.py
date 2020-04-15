import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,0.01,0.03,0.05,0.11,0.17])

y_FLIR = np.array([8.152,16.294,19.736,21.23,23.228,24.166])
e_FLIR = np.array([0.2143,0.3702,0.2625,0.2604,0.2450,0.2439])

y_LTIR = np.array([8.5092,16.494,20.372,22.48,24.258,25.212])
e_LTIR = np.array([0.1149,1.0405,1.0474,0.2544,0.8233,0.7177])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('Gaussian noise sigma')
plt.ylabel('noise visibility')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of noise visibility measure')

plt.show()
