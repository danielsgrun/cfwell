import cfwell.representations as rp
import cfwell.hamiltonian as ht
import cfwell.dynamics as dn

from math import pi, sqrt
from numpy import linspace

M,P = 4,11 # Initial state populations
ket0 = [M,P,0,0] # Initial state representation
natom = M+P # total number of atoms

U,J = 105, 72 # integr. hamiltonian parameters
nu,mu = 0,0 # breaking-of-int. parameters

tm = 2*pi*U/(J*J) * ((M-P)**2 - 1)

t0, tf = 0, 2*tm # initial time / final time
nt = 300 # number of time steps

time = linspace(t0, tf, num=nt)

params = {'interaction': U,
          'hopping': J,
          'break13': nu,
          'break24': mu}

Hint, Htun, Hb13, Hb24, H = ht.make_h(params, natom) # represents the hamilt.

# %% Quantum dynamics

Ni = rp.make_Ni(natom) # represents the Ni operators as matrices

psi0 = rp.representation(ket0) # represents the initial state in a vector

psit = dn.time_evol(psi0, H, time) # time-evolves the init. state
                                   # w/ the hamilt. provided

n1t, n2t, n3t, n4t = dn.quantum_dyn(Ni,            # calculates <n_i(t)> with    
                                    time,          # the time-evolved states 
                                    psit,          # provided.  
                                    natom,         
                                    square=False,
                                    plot=True)     

n12t, n22t, n32t, n42t = dn.quantum_dyn(Ni,
                                        time,
                                        psit,
                                        natom,
                                        square=True, # enables Ni@Ni, which
                                        plot=True)   # results in:
                                                     # <n_i²(t)>

# %% NOON state fidelity with respect to time

from numpy import array, dot, conj

coefs = 1./2. * array([1, 1, 1, -1]) # superposition coefficients
kets = array([[M,P,0,0], [M,0,0,P], [0,P,M,0], [0,0,M,P]]) # states in superp.

# WARNING: each coefficient is related to each of the "kets" in the same order!

NOON = rp.make_state(coefs, kets)

fids = abs(array([dot(conj(NOON), psit[i,:]) for i in range(len(time))]))

#%% Energy bands 

H_struc = [Hint, Htun, Hb13, Hb24] # hamiltonian matrix structures
ujlist, bvals = ht.bands(H_struc,
                         params, 
                         natom,
                         plot=True) # energy bands calculation

