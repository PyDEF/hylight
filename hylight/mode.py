"""Vibrational mode and related utilities.
"""

# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
from typing import Iterable, Optional, Union
import logging

import numpy as np

from .constants import eV_in_J, atomic_mass, hbar_si, two_pi, cm1_in_J
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
        self.delta = np.sqrt(1.0 / self.masses.reshape((-1, 1))) * eigenvector
        self.mass = np.sum(np.sum(self.eigenvector**2, axis=1) * self.masses)

    @property
    def energy_si(self):
        "Energy of the mode in J."
        return self.energy

    @property
    def energy_eV(self):
        "Energy of the mode in eV."
        return self.energy / eV_in_J

    @property
    def energy_meV(self):
        "Energy of the mode in meV."
        return self.energy / eV_in_J * 1e3

    @property
    def energy_cm1(self):
        "Energy of the mode in :math:`cm^{-1}`."
        return self.energy / cm1_in_J

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
        self.eigenvector = self.delta * np.sqrt(self.masses.reshape((-1, 1)))

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
            ) * amplitude * self.delta * np.sqrt(atomic_mass)
            traj.append(Atoms(self.atoms, coords))

        return traj

    def to_jmol(self, dest, **opts):
        """Write a mode into a Jmol file.

        See :py:func:`hylight.jmol.export` for the parameters.
        """
        from .jmol import export

        return export(dest, self, **opts)

    def participation_ratio(self):
        r"""Fraction of atoms active in the mode.

        R J Bell et al 1970 J. Phys. C: Solid State Phys. 3 2111
        https://doi.org/10.1088/0022-3719/3/10/013

        It is equal to :math:`M_1^2 / (M_2 M_0)` where

        .. math::

            M_n = \sum_\alpha {m_\alpha {||\eta_\alpha||}^2}^n

        where :math:`\eta_\alpha` is the contribution of atom :math:`\alpha` to
        eigendisplacement :math:`\eta`.

        But :math:`M_0 = N` by definition and :math:`M_1 = 1` because the eigenvectors are normalized.

        .. Note::

            :math:`M_n` = :code:`np.sum(self.energies()**n)`
        """

        return 1.0 / (
            np.sum(np.sum(self.eigenvector**2, axis=1) ** 2) * len(self.eigenvector)
        )

    def per_species_n_eff(self, sp):
        "Compute the number of atoms participating to the mode for a given species."
        block = self.eigenvector[np.array(self.atoms) == sp, :]

        if block.size == 0:
            raise ValueError(f"{sp} is absent from this structure.")

        if block.size == 1:
            raise ValueError(
                f"There is only one {sp}, participation ratio is always one."
            )

        return np.sum(block**2) ** 2 / np.sum(np.sum(block**2, axis=1) ** 2)

    def per_species_pr(self, sp):
        """Compute the fraction of atoms participation to the mode for a given species.

        See also :func:`per_species_n_eff`.
        """

        block = self.eigenvector[np.array(self.atoms) == sp, :]

        if block.size == 0:
            raise ValueError(f"{sp} is absent from this structure.")

        if block.size == 1:
            raise ValueError(
                f"There is only one {sp}, participation ratio is always one."
            )

        return np.sum(block**2) ** 2 / (
            np.sum(np.sum(block**2, axis=1) ** 2) * len(block)
        )

    def localization_ratio(self):
        """Quantify the localization of the mode over the supercell.

        See also :func:`participation_ratio`.

        1: fully delocalized
        >> 1: localized
        """

        return np.sum(np.sum(self.eigenvector**2, axis=1) ** 2) * len(self.eigenvector)

    def energies(self):
        "Return the energy participation of each atom to the mode."

        return np.sum(self.eigenvector**2, axis=1)


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
    """Compute vibrational modes from the dynamical matrix.

    :param lattice: lattice parameters (3 x 3 :class:`numpy.ndarray`)
    :param atoms: list of atoms
    :param masses: list of atom masses
    :param ref: reference position
    :param dynmat: dynamical matrix
    :returns: list of :class:`hylight.mode.Mode`
    """
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
    "A falsy value explaining how the cell are not matching."

    def __init__(self, reason, details):
        self.reason = reason
        self.details = details

    def __str__(self):
        return f"{self.reason}: {self.details}"

    def __bool__(self):
        return False


def same_cell(cell1: FArray, cell2: FArray, tol=1e-6) -> Union[CellMismatch, bool]:
    "Compare two lattice vectors matrix and return True if they describe the same cell."

    if (
        np.max(np.abs(np.linalg.norm(cell1, axis=1) - np.linalg.norm(cell2, axis=1)))
        > tol
    ):
        return CellMismatch(
            "length", np.linalg.norm(cell1, axis=1) - np.linalg.norm(cell2, axis=1)
        )

    a1: FArray
    a1, b1, c1 = cell1
    a2, b2, c2 = cell2

    if (
        np.max(
            np.abs(
                [
                    angle(a1, b1) - angle(a2, b2),
                    angle(b1, c1) - angle(b2, c2),
                    angle(c1, a1) - angle(c2, a2),
                ]
            )
        )
        > tol
    ):
        return CellMismatch(
            "length",
            [
                angle(a1, b1) - angle(a2, b2),
                angle(b1, c1) - angle(b2, c2),
                angle(c1, a1) - angle(c2, a2),
            ],
        )

    return True


def angle(v1, v2):
    "Compute the angle in radians between two 3D vectors."

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


def get_energies(phonons: Iterable[Mode], mask: Optional[Mask] = None) -> FArray:
    """Return an array of mode energies in SI."""
    if mask:
        return np.array(
            [ph.energy for ph in phonons if ph.real if mask.accept(ph.energy)]
        )
    else:
        return np.array([ph.energy for ph in phonons if ph.real])


def project_on_asr(mat, masses):
    """Enforce accoustic sum rule.

    Project the input matrix on the space of ASR abiding matrices and return
    the projections.
    """
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
