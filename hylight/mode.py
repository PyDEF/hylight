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
        self.delta = delta  # vibrational mode normalized

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

    def huang_rhys(self, delta_R, use_q=False):
        r"""Compute the Huang-Rhyes factor

        :param delta_R: displacement in SI
        :param use_q:
          true:  S_i = 1/2 \frac{\omega}{\hbar}   {\Delta Q_i}^2
          false: S_i = 1/2 \frac{\omega}{\hbar} m {\Delta R_i}^2
        """

        if use_q is True or use_q == CoordMode.Hybrid:
            # Formula from Alkauskas et. al. New J. Phys., 2014, equations (5) and (6)
            delta_Q_i = np.sqrt(self.masses).dot(np.sum(self.delta * delta_R, axis=1))
            return 0.5 * self.energy / hbar_si ** 2 * delta_Q_i ** 2
        elif use_q is False and use_q == CoordMode.UseR:
            delta_R_i_2 = self.project_coef2(delta_R)  # in SI
            return 0.5 * self.mass * self.energy / hbar_si ** 2 * delta_R_i_2
        elif use_q == CoordMode.ComputeQ:
            delta_Q = np.sqrt(self.masses) * delta_R
            delta_Q_i_2 = self.project_coef2(delta_Q)  # in SI
            return 0.5 * self.energy / hbar_si ** 2 * delta_Q_i_2
        else:
            raise ValueError(f"Unexpected value for use_q: {use_q}")

    def to_traj(self, duration, amplitude):
        """Produce a ase trajectory for animation purpose.

        :param duration: duration of the animation in seconds (framerate is 25)
        :param amplitude: amplitude applied to the mode in A (the modes are normalized)
        """
        from ase import Atoms

        n = int(duration * 25)

        traj = []
        for i in range(n):
            coords = self.ref + np.sin(two_pi * i / n) * amplitude * self.delta
            traj.append(Atoms(self.atoms, coords))
        return traj
