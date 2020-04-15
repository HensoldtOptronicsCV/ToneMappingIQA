import matplotlib.pyplot as plt
import numpy as np

x_FLIR = np.array([0,0.05,0.1,0.15,0.198,0.248,0.298])
y_FLIR = np.array([0.0000116,0.0083,0.0116,0.021,0.0264,0.036,0.039])
e_FLIR= np.array([0,0.003,0.01,0.0124,0.024,0.02,0.03])

x_LTIR = np.array([0,0.05,0.1,0.15,0.202,0.252,0.302])
y_LTIR = np.array([0.0001,0.0005,0.00096,0.0014,0.0021,0.0076,0.0093])
e_LTIR= np.array([0.00001,0.0002,0.0002,0.001,0.004,0.017,0.01])

plt.errorbar(x_FLIR, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x_LTIR, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('global temporal incoherence')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of global temporal incoherence measure')

plt.show()
