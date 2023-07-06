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
from typing import Iterable, Optional, Union
import logging

import numpy as np

from .constants import eV_in_J, atomic_mass, hbar_si, two_pi
from .typing import FArray, BArray


log = logging.getLogger("hylight")


class Mode:
    """The representation of a vibrational mode.

    It stores the eigenvector and eigendisplacement.
    It can be used to project other displacement on the eigenvector.
    """

    def __init__(
        self,
        lattice: FArray,
        atoms: list[str],
        n: int,
        real: bool,
        energy: float,
        ref: FArray,
        eigenvector: FArray,
        masses: Iterable[float],
    ):
        """Build the mode from OUTCAR data.

        :param atoms: list of atoms
        :param n: numeric id in OUTCAR
        :param real: boolean, has the mode a real frequency ?
        :param energy: energy/frequency of the mode, expected in meV
        :param ref: equilibrium position of atoms
        :param delta: displacement array np.ndarray((natoms, 3), dtype=float)
        :param masses: masses of the atoms in atomic unit
        """
        self.lattice = lattice
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

    def set_lattice(self, lattice: FArray, tol=1e-6) -> None:
        """Change the representation to another lattice.

        :param lattice: 3x3 matrix representing lattice vectors :code:`np.array([a, b, c])`.
        :param tol: numerical tolerance for vectors mismatch.
        """

        if self.lattice is None:
            log.warning(
                "Lattice was previously unkown. Assuming it was the same as the one provided now."
            )
            self.lattice = lattice
            return

        sm = same_cell(lattice, self.lattice, tol=tol)
        if not sm:
            raise ValueError(
                "The new lattice vectors describe a different cell from the previous one."
                f"\n{sm}"
            )

        self.delta = self.delta @ np.linalg.inv(self.lattice) @ lattice
        self.ref = self.ref @ np.linalg.inv(self.lattice) @ lattice
        self.lattice = lattice
        self.eigenvector = self.delta / np.sqrt(self.masses.reshape((-1, 1)))

    def project(self, delta_Q: FArray) -> float:
        """Project delta_Q onto the eigenvector."""
        delta_R_dot_mode = np.sum(delta_Q * self.eigenvector)
        return delta_R_dot_mode * self.eigenvector

    def project_coef2(self, delta_Q: FArray) -> float:
        """Square lenght of the projection of delta_Q onto the eigenvector."""
        delta_Q_dot_mode = np.sum(delta_Q * self.eigenvector)
        return delta_Q_dot_mode**2

    def project_coef2_R(self, delta_R: FArray) -> float:
        """Square lenght of the projection of delta_R onto the eigenvector."""
        delta_Q = np.sqrt(self.masses).reshape((-1, 1)) * delta_R
        return self.project_coef2(delta_Q)

    def huang_rhys(self, delta_R: FArray) -> float:
        r"""Compute the Huang-Rhyes factor.

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
            coords = self.ref + np.sin(
                two_pi * i / n
            ) * amplitude * self.delta / np.sqrt(atomic_mass)
            traj.append(Atoms(self.atoms, coords))

        return traj

    def to_jmol(self, dest, **opts):
        """Write a mode into a Jmol file.

        See :py:func:`hylight.jmol.export` for the parameters.
        """
        from .jmol import export

        return export(dest, self, **opts)


def rot_c_to_v(phonons: Iterable[Mode]) -> FArray:
    """Rotation matrix from Cartesian basis to Vibrational basis (right side)."""
    return np.array([m.eigenvector.reshape((-1,)) for m in phonons])


def dynamical_matrix(phonons: Iterable[Mode]) -> FArray:
    """Retrieve the dynamical matrix from a set of modes.

    Note that if some modes are missing the computation will fail.

    :param phonons: list of modes
    """
    dynamical_matrix_diag = np.diag(
        [(1 if m.real else -1) * (m.energy / hbar_si) ** 2 for m in phonons]
    )
    Lt = rot_c_to_v(phonons)

    return Lt.transpose() @ dynamical_matrix_diag @ Lt


def modes_from_dynmat(lattice, atoms, masses, ref, dynmat):
    n, _ = dynmat.shape
    assert n % 3 == 0
    n_atoms = n // 3
    # eigenvalues and eigenvectors, aka square of angular frequencies and normal modes
    vals, vecs = np.linalg.eigh(dynmat)
    vecs = vecs.transpose()
    assert vecs.shape == (n, n)

    # modulus of mode energies in J
    energies = hbar_si * np.sqrt(np.abs(vals)) / eV_in_J * 1e3

    # eigenvectors reprsented in canonical basis
    vx = vecs.reshape((n, n_atoms, 3))

    return [
        Mode(
            lattice,
            atoms,
            i,
            e2 >= 0,  # e2 < 0 => imaginary frequency
            e,
            ref,
            v.reshape((-1, 3)),
            masses,
        )
        for i, (e2, e, v) in enumerate(zip(vals, energies, vx))
    ]

class Mask:
    "An energy based mask for the set of modes."

    def __init__(self, intervals: list[tuple[float, float]]):
        self.intervals = intervals

    @classmethod
    def from_bias(cls, bias: float) -> "Mask":
        """Create a mask that reject modes of energy between 0 and `bias`.

        :param bias: minimum of accepted energy (eV)
        :returns: a fresh instance of `Mask`.
        """
        if bias > 0:
            return cls([(0, bias * eV_in_J)])
        else:
            return cls([])

    def add_interval(self, interval: tuple[float, float]) -> None:
        "Add a new interval to the mask."
        assert (
            isinstance(interval, tuple) and len(interval) == 2
        ), "interval must be a tuple of two values."
        self.intervals.append(interval)

    def as_bool(self, ener: FArray) -> BArray:
        "Convert to a boolean `np.ndarray` based on `ener`."
        bmask = np.ones(ener.shape, dtype=bool)

        for bot, top in self.intervals:
            bmask &= (ener < bot) | (ener > top)

        return bmask

    def accept(self, value: float) -> bool:
        "Return True if value is not under the mask."
        return not any(bot <= value <= top for bot, top in self.intervals)

    def reject(self, value: float) -> bool:
        "Return True if `value` is under the mask."
        return any(bot <= value <= top for bot, top in self.intervals)

    def plot(self, ax, unit):
        """Add a graphical representation of the mask to a plot.

        :param ax: a matplotlib `Axes` object.
        :param unit: the unit of energy to use (ex: :attr:`hylight.constant.eV_in_J` if the plot uses eV)
        :returns: a function that must be called without arguments after resizing the plot.
        """
        from matplotlib.patches import Rectangle

        rects = []
        for bot, top in self.intervals:
            p = Rectangle((bot / unit, 0), (top - bot) / unit, 0, facecolor="grey")
            ax.add_patch(p)
            rects.append(p)

        def resize():
            (_, h) = ax.transAxes.transform((0, 1))
            for r in rects:
                r.set(height=h)

        return resize


class CellMismatch:
    def __init__(self, reason, details):
        self.reason = reason
        self.details = details

    def __str__(self):
        return f"{self.reason}: {self.details}"

    def __bool__(self):
        return False


def same_cell(cell1: FArray, cell2: FArray, tol=1e-6) -> Union[CellMismatch, bool]:
    "Compare two lattice vectors matrix and return True if they describe the same cell."

    if np.max(np.abs(np.linalg.norm(cell1, axis=1) - np.linalg.norm(cell2, axis=1))) > tol:
        return CellMismatch("length", np.linalg.norm(cell1, axis=1) - np.linalg.norm(cell2, axis=1))

    a1: FArray
    a1, b1, c1 = cell1
    a2, b2, c2 = cell2

    if np.max(
        np.abs(
            [
                angle(a1, b1) - angle(a2, b2),
                angle(b1, c1) - angle(b2, c2),
                angle(c1, a1) - angle(c2, a2),
            ]
        )
    ) > tol :
        return CellMismatch("length", [
                angle(a1, b1) - angle(a2, b2),
                angle(b1, c1) - angle(b2, c2),
                angle(c1, a1) - angle(c2, a2),
            ])

    return True


def angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v1 / np.linalg.norm(v2)
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), v1.dot(v2))


def get_HR_factors(
    phonons: Iterable[Mode], delta_R_tot: FArray, mask: Optional[Mask] = None
) -> FArray:
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


def get_energies(phonons: Iterable[Mode], mask: Optional[Mask]=None) -> FArray:
    """Return an array of mode energies in SI."""
    if mask:
        return np.array(
            [ph.energy for ph in phonons if ph.real if mask.accept(ph.energy)]
        )
    else:
        return np.array([ph.energy for ph in phonons if ph.real])


def project_on_asr(mat, masses):
    n, *_ = mat.shape
    assert mat.shape == (n, n), "Not a square matrix."
    assert n % 3 == 0, "Matrix size is not 3n."

    basis = np.eye(n)

    masses = [(m * atomic_mass) for m in masses]

    # this is the configurational displacement that correspond to a rigid
    # displacement of atoms
    m = np.sqrt(masses / np.sum(masses))
    basis[0, 0::3] = m
    basis[1, 1::3] = m
    basis[2, 2::3] = m

    orthonormalize(basis, n_skip=3)

    # Projector in the adapted basis
    proj = np.eye(n)
    proj[0, 0] = proj[1, 1] = proj[2, 2] = 0.0

    return mat @ basis @ proj @ basis.T


def generate_basis(seed):
    """Generate an orthonormal basis with the rows of seed as first rows.

    :param seed: the starting vectors, a :code:`(m, n)` matrix of orthonormal rows.
        :code:`m = 0` is valid and will create a random basis.
    :return: a :code:`(n, n)` orthonormal basis where the first :code:`m` rows
        are the rows of :code:`seed`.
    """

    assert np.allclose(
        np.eye(len(seed)), seed @ seed.T
    ), "Seed is not a set of orthonormal vectors."

    n_seed, n = seed.shape
    m = np.zeros((n, n))

    m[:n_seed, :] = seed

    for i in range(n_seed, n):
        prev = m[: i - 1, :]

        res = 1
        c = np.random.uniform(size=(1, n))

        # poorly conditioned initial condition can lead to numerical errors
        # in the orthonormalisation.
        # This loop will ensure that the new vector is mostly orthogonal to all
        # its predecessor.
        # It does more iterations for later vectors as one would expect.
        # It should be less than a 100 iterations in worse case.
        while not np.allclose(res, 0):
            res = (c @ prev.T) @ prev
            c -= res
            c /= np.linalg.norm(c)

        m[i, :] = c

    # We still need to polish the orthonormalisation to reach machine precision
    # limit. Fortunatly the initial guess is good enough that the loop in
    # orthogonalize will only iterate 3 to 5 times even for large matrices
    orthonormalize(m, n_skip=n_seed)

    return m


def orthonormalize(m, n_skip=0):
    """Ensure that the vectors of m are orthonormal.

    Change the rows from n_seed up inplace to make them orthonormal.

    :param m: the starting vectors
    :param n_seed: number of first rows to not change.
        They must be orthonormal already.
    """

    (n, _) = m.shape
    assert m.shape == (n, n), "m is not a square matrix"

    if n_skip > 0:
        s = m[:n_skip, :]
        assert np.allclose(
            np.eye(n_skip), s @ s.T
        ), "Seed is not a set of orthonormal vectors."

    eye = np.eye(n)

    while not np.allclose(eye, m @ m.T):
        for i in range(n_skip, n):
            # get the matrix without row i 
            rest = m[[j for j in range(n) if j != i], :]
            c = m[i, :]
            # remove the part of c that is in the subspace of rest
            c -= (c @ rest.T) @ rest
            # renormalize
            c /= np.linalg.norm(c)
            m[i, :] = c

    return m
