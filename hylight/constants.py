h_si = 6.62607015e-34  # SI (J.s)
c_si = 2.99792e8  # SI (m.s-1)
h_c = h_si * c_si  # SI (J.m)
eV_in_J = 1.602176634e-19
two_pi = 6.283185307179586
pi = 3.141592653589793
hbar_si = h_si / two_pi  # J.s.rad-1
hbar = h_si / two_pi / eV_in_J  # ev.s.rad-1

atomic_mass = 1.67e-27  # SI

THz_in_meV = 4.135665538536

cm1_in_J = 100 * h_c

kb = 1.38064852e-23  # J.K-1 Boltzmann constant
kb_eV = kb / eV_in_J  # eV.K-1
