import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45])

y_FLIR = np.array([-0.042687178, -0.03786037, -0.032523904, -0.027373618, -0.023536311, -0.019857196, -0.015819354, -0.011666357, -0.0072056125, -0.0024240182])
e_FLIR = np.array([0.0046185777, 0.0055545415, 0.0050877924, 0.0047490806, 0.00458089, 0.0042861523, 0.0038656208, 0.0032853552, 0.0024428181, 0.0013989488])

y_LTIR = np.array([-0.02283526, -0.020658439, -0.018370962, -0.016113963, -0.013751363, -0.011327258, -0.008877209, -0.0063813226, -0.0038422316, -0.0012987119])
e_LTIR = np.array([0.0096077295, 0.008384871, 0.0072270436, 0.006114621, 0.00506374, 0.0040571955, 0.0031071298, 0.0022403877, 0.0015602956, 0.0011039249])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('local loss of contrast')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of local loss of contrast measure')

plt.show()
