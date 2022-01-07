import numpy as np
import scipy.fft as fft

from .loader import load_phonons, load_poscar
from .constants import h_si, two_pi, eV_in_J, THz_in_meV, atomic_mass, hbar_si


def spectra(
    outcar, poscar_gs, poscar_es, zpl, sigma=1e-5, resolution_e=1e-5, N=1_000_000
):
    """
    outcar, poscar_es, poscar_gs are path
    zpl is in eV
    sigma is in s-1 ?
    """

    phonons, _, masses = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    pl = Photoluminescence(phonons, masses, 1000, 1e-6)
    pl.D_R = delta_R

    pl.compute()
    _, I = pl.PL(sigma, 0, zpl)
    return pl.omega_set, np.array(I), pl.S_omega


def compute_delta_R(poscar_gs, poscar_es):
    """Return $\\Delta R$ in A."""
    return load_poscar(poscar_gs) - load_poscar(poscar_es)


class Photoluminescence:

    def __init__(self, phonons, masses, resolution, S_disp=6e-3):
        self.resolution = resolution
        self.S_disp = S_disp
        self.m = []
        self.line_shape = "gaussian"

        self.numAtoms = len(masses)

        self.HuangRhyes = 0
        self.Delta_R = 0
        self.Delta_Q = 0
        self.IPR = []
        self.q = []
        self.r = []
        self.S = []
        self.phonons = phonons
        self.frequencies = [0 for _ in phonons]
        self.masses = masses
        self.D_R = None

    def compute(self):
        for i, ph in enumerate(self.phonons):
            q_i = 0
            r_i = 0
            IPR_i = 0
            participation = 0
            self.frequencies[i] = ph.energy / eV_in_J

            if not ph.real:
                self.frequencies[i] = 0

            max_Delta_r = 0

            for a in range(self.numAtoms):
                # Normalize r:
                participation = (
                    ph.delta[a, 0]**2
                    + ph.delta[a, 1]**2
                    + ph.delta[a, 2]**2
                )
                IPR_i += participation**2

                for coord in range(3):
                    q_i += (
                        np.sqrt(self.masses[a])
                        * (self.D_R[a, coord])
                        * ph.delta[a, coord]
                        * 1e-10
                    )
                    r_i += (
                        self.D_R[a, coord]
                        * ph.delta[a, coord]
                        * 1e-10
                    )

                    if np.abs(ph.delta[a, coord]) > max_Delta_r:
                        max_Delta_r = np.abs(ph.delta[a, coord])

            IPR_i = 1.0 / IPR_i
            S_i = self.frequencies[i] * q_i**2 / 2 * 1.0 / \
                (1.0545718e-34 * 6.582119514e-16)

            self.IPR += [IPR_i]
            self.q += [q_i]
            self.r += [r_i]
            self.S += [S_i]
            self.HuangRhyes += S_i

        for a in range(self.numAtoms):
            for coord in range(3):
                self.Delta_R += (self.D_R[a][coord])**2
                self.Delta_Q += (self.D_R[a][coord])**2 * self.masses[a]

        self.Delta_R = self.Delta_R**0.5

        self.Delta_Q = (self.Delta_Q / 1.660539040e-27) ** 0.5

        self.max_energy = 5

        self.omega_set = np.linspace(
            0, self.max_energy, int(self.max_energy*self.resolution))
        self.S_omega = [self.get_S_omega(o, self.S_disp) for o in self.omega_set]

    def get_S_omega(self, omega, sigma):
        sum = 0
        for k in range(len(self.S)):
            sum += self.S[k] * self.gaussian(omega, self.frequencies[k], sigma)
        return sum

    def gaussian(self, omega, omega_k, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(omega - omega_k) * (omega - omega_k) / sigma / sigma / 2)

    '''
    The actual photoluminescence line-shape calculation:
    Calculate the line-shape function after reading the calculated vibrational data
    '''
    def PL(self, gamma, SHR, EZPL):
        Gt = []  # G with lorentizian shape line applied
        I = []  #The PL intensity function

        r = 1/self.resolution
        St = fft.ifft(self.S_omega)
        St = fft.ifftshift(St)  # The Fourier-transformed partial HR function
        G = np.exp(2*np.pi*St-SHR)

        for i in range(len(G)):
            t = r*(i-len(G)/2)
            if self.line_shape == "gaussian":
                Gt += [G[i] * self.gaussian(t, 0, 1/gamma)]
            elif self.line_shape == "lorentzian":
                Gt += [G[i] * np.exp(-gamma * abs(t))]
            else:
                raise NotImplementedError(f"\"{self.line_shape}\" line shape is not implemented.")

        A = fft.fft(Gt)

        # Now, shift the ZPL peak to the EZPL energy value
        tA = A.copy()
        for i in range(len(A)):
            A[(int(EZPL*self.resolution)-i) % len(A)] = tA[i]

        for i in range(len(A)):
            I += [A[i]*((i)*r)**3]

        return A, np.array(I)
