from .constants import eV_in_J, atomic_mass, h_si, hbar_si, two_pi
import numpy as np


class Mode:
    def __init__(self, atoms, n, real, energy, ref, delta, masses):
        self.atoms = atoms
        self.n = n  # numeric id in VASP
        self.real = real  # False if imaginary coordinates
        self.energy = energy * 1e-3 * eV_in_J  # energy from meV to SI
        self.ref = ref  # equilibrium position in A
        self.delta = delta  # vibrational mode normalized

        self.masses = np.array(masses) * atomic_mass

        # effective mass in kg
        self.mass = (np.linalg.norm(self.delta, axis=1) ** 2).dot(self.masses)

    def project(self, delta_R):
        delta_R_dot_mode = np.sum(delta_R * self.delta)
        return delta_R_dot_mode * self.delta

    def project_coef2(self, delta_R):
        delta_R_dot_mode = np.sum(delta_R * self.delta)
        return delta_R_dot_mode ** 2

    def huang_rhys(self, delta_R_tot, use_q=False):
        """
        delta_R_tot in SI
        """
        delta_R_i_2 = self.project_coef2(delta_R_tot)  # in SI

        if use_q:
            delta_Q_i = np.sqrt(self.masses).dot(np.sum(self.delta * delta_R_tot, axis=1))
            return 0.5 * self.energy / hbar_si ** 2 * delta_Q_i**2
        else:
            return 0.5 * self.mass * self.energy / hbar_si ** 2 * delta_R_i_2

    def to_traj(self, duration, amplitude):
        from ase import Atoms

        n = int(duration * 25)

        traj = []
        for i in range(n):
            coords = self.ref + np.sin(two_pi * i / n) * amplitude * self.delta
            traj.append(Atoms(self.atoms, coords))
        return traj
