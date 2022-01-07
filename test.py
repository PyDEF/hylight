import numpy as np
import matplotlib.pyplot as plt


from hylight.mp import spectra #, debug_thing


nu, I, s = spectra("/mnt/these/bazro3/phonons/OUTCAR",
                "/mnt/these/bazro3/phonons/POSCAR",
                "/mnt/these/bazro3/phonons/POSCAR_ES", 2.97)

plt.plot(nu, s)
plt.figure()
n = len(nu)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
ax1.sharex(ax2)
ax1.plot(nu, I.real, label="real")
ax2.plot(nu, I.imag, label="imag")
plt.legend()

plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
ax1.sharex(ax2)
ax1.plot(nu, np.abs(I), label="magnitude")
ax2.plot(nu, np.angle(I), label="phase")
plt.legend()
plt.show()

# locals().update(debug_thing[0])
