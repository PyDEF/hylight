import numpy as np
import matplotlib.pyplot as plt

from hylight.loader import load_phonons
from hylight.mono_phonon import compute_spectra as mono_spectra

from hylight.multi_phonons import (
    compute_spectra as mul_spectra,
    compute_delta_R,
    hr_spectra,
    fc_spectra,
)


prefix = "../test_bzo/"

phonons, _, _ = load_phonons(prefix + "OUTCAR")
delta_R = compute_delta_R(prefix + "POSCAR", prefix + "POSCAR_ES")

zpl = 2.97

nu_mul, I_mul = mul_spectra(
    phonons,
    delta_R,
    zpl,
    sigma=-100,
    resolution_e=1e-3,
    e_max=8,
)

I_mul /= np.max(np.abs(I_mul))

nu_mono, I_mono, _ = mono_spectra(
    zpl,
    300,
    0.2871,
    0.1091,
    83.14e-3,
    e_max=8,
)

f, fc, dirac_fc = fc_spectra(phonons, delta_R)
f, s, dirac_s = hr_spectra(phonons, delta_R)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.stackplot(f, s, color="grey")
ax1.plot(f, dirac_s, color="black")
ax1.set_ylabel("$S(\\hbar\\omega)$ (A. U.)")

ax2.stackplot(f, fc, color="grey")
ax2.plot(f, dirac_fc, color="black")
ax2.set_ylabel("$FC shift$ (A. U.)")
ax2.set_xlabel("Phonon energy (meV)")

plt.figure()
plt.plot(nu_mul, np.abs(I_mul), label="Multi")
plt.plot(nu_mono, np.abs(I_mono), label="Mono")
plt.xlim(1, 5)
plt.legend()
plt.show()
