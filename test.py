import numpy as np
import matplotlib.pyplot as plt


from hylight.spectra import spectra, debug_thing


nu, I = spectra("../phonons/OUTCAR", "../phonons/POSCAR", "../phonons/POSCAR_ES", 2.97)

n = len(nu)
plt.plot(nu[:int(0.8*n)], I[:int(0.8*n)])
plt.show()

locals().update(debug_thing[0])
