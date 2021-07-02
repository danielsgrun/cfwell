from numpy import zeros, dot, conj, array,exp, diag, where
from math import sqrt
from tqdm import tqdm

def time_evol(psi0, H, time):
  '''
    

    Parameters
    ----------
    psi0 : np.array
        Representation of initial state.
    H : np.array
        Representation of >complete< hamiltonian.
    nt : np.array
        Array containing all time points.

    Returns
    -------
       psit

    '''  
 
  from scipy.linalg import eigh

  dim = len(psi0)
  nt = len(time)
  
  psit = zeros((nt, dim), dtype=complex)
  
  evl, evc = eigh(H)
                 
  for t in tqdm(range(0,len(time))):
    psit[t,:] = sum(dot(conj(evc[:,i]), psi0)*exp(-1j*evl[i]*(time[t]))*evc[:,i]
                    for i in range(0,dim))  
  
  return psit




def quantum_dyn(Ni, time, psit, natom, square=False, plot=True): 
  '''
    

    Parameters
    ----------
    Ni : np.array
        Array containing the representations of N1, N2, N3, N4 in this order:
            "np.array([N1, N2, N3, N4])"
    time : np.array
        Array containing the time points.
    psit : np.array
        Array containing the representations of |Psi(t)> for each t:
            [|Psi(0)>, |Psi(dt)>, ... |Psi(tf)>]
    natom : int
        Total # of atoms.
    square : bool, optional
        Decides between <n(t)> and <n²(t)>. The default is False.    
    plot : bool, optional
        Decides whether to plot or not. The default is True.

    Returns
    -------
    np.array([n1t, n2t, n3t, n4t])

    '''  
      
  nb = natom
  nt, dim = len(psit[:,0]), len(psit[0,:])
  n1t, n2t, n3t, n4t = (zeros((nt)),
                        zeros((nt)),
                        zeros((nt)),
                        zeros((nt)))

  if square==True:
    Ni = array([Ni[i]@Ni[i] for i in range(4)])
    labels = ["$<n_1^2>$",
              "$<n_2^2>$",
              "$<n_3^2>$",
              "$<n_4^2>$"]
  else:
    labels = ["$<n_1>$",
              "$<n_2>$",
              "$<n_3>$",
              "$<n_4>$"]  
    
  
  for i in tqdm(range(0,nt)):
    n1t[i], n2t[i], n3t[i], n4t[i] = (dot(conj(psit[i,:]), (Ni[0] @ psit[i,:])),
                                        dot(conj(psit[i,:]), (Ni[1] @ psit[i,:])),
                                        dot(conj(psit[i,:]), (Ni[2] @ psit[i,:])),
                                        dot(conj(psit[i,:]), (Ni[3] @ psit[i,:])))

  n1t, n2t, n3t, n4t = 1/nb*array([n1t, n2t, n3t, n4t])
            

  if plot==True:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11,8))
    plt.plot(time, n1t, label=labels[0], 
           lw=2.0, alpha=0.7, color='k')
    plt.plot(time, n2t, label=labels[1], 
           lw=2.0, alpha=0.7, color='cyan')
    plt.plot(time, n3t, label=labels[2], 
           lw=2.0, alpha=0.7, color='purple')
    plt.plot(time, n4t, label=labels[3], 
           lw=2.0, alpha=0.7, color='green')
    plt.legend(loc=0, fontsize=21)
    plt.xlabel("t(s)", fontsize=21)
    plt.ylabel("Exp. value", fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.show()

  return array([n1t, n2t, n3t, n4t])


def meas(vec, N, i):

  dimens = len(vec)
  coef03 = 0.
  psi03 = zeros((dimens), dtype=complex)

  for j in where(diag(N)==i)[0]: 
    coef03 += abs(vec[j])**2
    psi03[j] = vec[j]

  if coef03 != 0:
    psi03 = psi03 / sqrt(coef03)

  return [array([coef03]),
          array([psi03])]  
