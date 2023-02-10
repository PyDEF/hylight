"Vibrational mode and related utilities."
# License
#     Copyright (C) 2023  PyDEF development team
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from .constants import eV_in_J, atomic_mass, hbar_si, two_pi
import numpy as np


class Mode:
    """The representation of a vibrational mode.

    It stores the eigenvector and eigendisplacement.
    It can be used to project other displacement on the eigenvector.
    """

    def __init__(self, atoms, n, real, energy, ref, eigenvector, masses):
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
        self.eigenvector = eigenvector  # vibrational mode eigenvector (norm of 1)
        self.masses = np.array(masses) * atomic_mass
        # vibrational mode eigendisplacement
        self.delta = np.sqrt(self.masses.reshape((-1, 1))) * eigenvector
        self.mass = np.linalg.norm(self.delta) ** 2

    def project(self, delta_Q):
        """Project delta_Q onto the eigenvector"""
        delta_R_dot_mode = np.sum(delta_Q * self.eigenvector)
        return delta_R_dot_mode * self.eigenvector

    def project_coef2(self, delta_Q):
        """Square lenght of the projection of delta_Q onto the eigenvector."""
        delta_Q_dot_mode = np.sum(delta_Q * self.eigenvector)
        return delta_Q_dot_mode**2

    def project_coef2_R(self, delta_R):
        """Square lenght of the projection of delta_R onto the eigenvector."""
        delta_Q = np.sqrt(self.masses).reshape((-1, 1)) * delta_R
        return self.project_coef2(delta_Q)

    def huang_rhys(self, delta_R):
        r"""Compute the Huang-Rhyes factor

        .. math::
            S_i = 1/2 \frac{\omega}{\hbar} [({M^{1/2}}^T \Delta R) \cdot \gamma_i]^2
            = 1/2 \frac{\omega}{\hbar} {\sum_i m_i^{1/2} \gamma_i {\Delta R}_i}^2

        :param delta_R: displacement in SI
        """

        delta_Q = np.sqrt(self.masses).reshape((-1, 1)) * delta_R
        delta_Q_i_2 = self.project_coef2(delta_Q)  # in SI
        return 0.5 * self.energy / hbar_si**2 * delta_Q_i_2

    def to_traj(self, duration, amplitude, framerate=25):
        """Produce a ase trajectory for animation purpose.

        :param duration: duration of the animation in seconds
        :param amplitude: amplitude applied to the mode in A (the modes are normalized)
        :param framerate: number of frame per second of animation
        """
        from ase import Atoms

        n = int(duration * framerate)

        traj = []
        for i in range(n):
            coords = self.ref + np.sin(two_pi * i / n) * amplitude * self.delta
            traj.append(Atoms(self.atoms, coords))

        return traj

    def to_jmol(self, **opts):
        """Write a mode into a Jmol file.

        See :py:func:`hylight.jmol.export` for the parameters.
        """
        from .jmol import export

        return export(self, **opts)


def rot_c_to_v(phonons):
    """Rotation matrix from Cartesian basis to Vibrational basis (right side)."""
    return np.array([m.eigenvector.reshape((-1,)) for m in phonons])


def dynamical_matrix(phonons):
    """Retrieve the dynamical matrix from a set of modes.

    Note that if some modes are missing the computation will fail.

    :param phonons: list of modes
    """
    dynamical_matrix_diag = np.diag(
        [(1 if m.real else -1) * (m.energy / hbar_si) ** 2 for m in phonons]
    )
    Lt = rot_c_to_v(phonons)

    return Lt.transpose() @ dynamical_matrix_diag @ Lt


def get_HR_factors(phonons, delta_R_tot, mask=None):
    """Compute the Huang-Rhys factors for all the real modes with energy above bias.

    :param phonons: list of modes
    :param delta_R_tot: displacement in SI
    :param mask: a mask to filter modes based on their energies.
    """
    if mask:
        return np.array(
            [
                ph.huang_rhys(delta_R_tot)
                for ph in phonons
                if ph.real
                if mask.accept(ph.energy)
            ]
        )
    else:
        return np.array([ph.huang_rhys(delta_R_tot) for ph in phonons if ph.real])


def get_energies(phonons, mask=None):
    """Return an array of mode energies in SI"""
    if mask:
        return np.array(
            [ph.energy for ph in phonons if ph.real if mask.accept(ph.energy)]
        )
    else:
        return np.array([ph.energy for ph in phonons if ph.real])
