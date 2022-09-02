from .constants import eV_in_J, atomic_mass, h_si, hbar_si, two_pi
import numpy as np
from enum import Enum, auto


class CoordMode(Enum):
    Hybrid = True
    UseR = False
    ComputeQ = auto()


class Mode:
    """The representation of a vibrational mode."""

    def __init__(self, atoms, n, real, energy, ref, delta, masses):
        """Build the mode from OUTCAR data.

        :param atoms: list of atoms
        :param n: numeric id in OUTCAR
        :param real: boolean, has the mode a real frequency ?
        :param energy: energy/frequency of the mode, expected in meV
        :param ref: equilibrium position of atoms
        :param delta: displacement array np.ndarray((natoms, 3), dtype=float)
        :param masses: masses of the atoms in atomic unit
        """
        self.atoms = atoms
        self.n = n  # numeric id in VASP
        self.real = real  # False if imaginary coordinates
        self.energy = energy * 1e-3 * eV_in_J  # energy from meV to SI
        self.ref = ref  # equilibrium position in A
        self.delta = delta  # vibrational mode eigenvector (norm of 1)

        self.masses = np.array(masses) * atomic_mass

        # effective mass in kg
        self.mass = (np.linalg.norm(self.delta, axis=1) ** 2).dot(self.masses)

    def project(self, delta_R):
        """Project delta_R onto the mode"""
        delta_R_dot_mode = np.sum(delta_R * self.delta)
        return delta_R_dot_mode * self.delta

    def project_coef2(self, delta_R):
        """Square lenght of the projection of delta_R onto the mode."""
        delta_R_dot_mode = np.sum(delta_R * self.delta)
        return delta_R_dot_mode ** 2

    def huang_rhys(self, delta_R):
        r"""Compute the Huang-Rhyes factor

        :param delta_R: displacement in SI
        S_i = 1/2 \frac{\omega}{\hbar} [({M^{1/2}^T \Delta R) \cdot \gamma_i]^2
        = 1/2 \frac{\omega}{\hbar} {\sum_i m_i^{1/2} \gamma_i {\Delta R}_i}^2
        """

        delta_Q = np.sqrt(self.masses).reshape((-1, 1)) * delta_R
        delta_Q_i_2 = self.project_coef2(delta_Q)  # in SI
        return 0.5 * self.energy / hbar_si ** 2 * delta_Q_i_2

    def to_traj(self, duration, amplitude, framerate=25):
        """Produce a ase trajectory for animation purpose.

        :param duration: duration of the animation in seconds
        :param amplitude: amplitude applied to the mode in A (the modes are normalized)
        :param framerate: (optional, default 25) number of frame per second of animation
        """
        from ase import Atoms

        n = int(duration * framerate)

        traj = []
        for i in range(n):
            coords = self.ref + np.sin(two_pi * i / n) * amplitude * self.delta
            traj.append(Atoms(self.atoms, coords))

        return traj


def get_HR_factors(phonons, delta_R_tot, bias=0):
    """
    delta_R_tot in SI
    """
    return np.array([ph.huang_rhys(delta_R_tot)
                     for ph in phonons
                     if ph.real
                     if ph.energy >= bias])


def get_energies(phonons, bias=0):
    """Return an array of mode energies in SI"""
    return np.array([ph.energy
                     for ph in phonons
                     if ph.real
                     if ph.energy >= bias])
