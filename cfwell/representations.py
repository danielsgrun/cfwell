from numpy import zeros
import numpy as np

def make_Ni(natom):
  '''
    

    Parameters
    ----------
    natom : int
        Total # of atoms.

    Returns
    -------
    np.array
        Array containing the representations of Ni operators.
        "np.array([N1, N2, N3, N4])."

    ''' 
    
  
  from numpy import array
  
  nb = natom
  dim = int((nb+3)*(nb+2)*(nb+1)/6)  
    
  N1, N2, N3, N4 = (zeros((dim,dim)),
                      zeros((dim,dim)),
                      zeros((dim,dim)),
                      zeros((dim,dim)))
    
  s = 0
    
  for i1 in range(0,nb+1):
    for j1 in range(0,nb+1-i1):
      for k1 in range(0,nb+1-i1-j1):
        l1 = nb - i1 - j1 - k1
        ss = 0
        for i2 in range(0,nb+1):
          for j2 in range(0,nb+1-i2):
            for k2 in range(0,nb+1-i2-j2):
              l2 = nb - k2 - j2 - i2
                
              N1[ss,s] = i1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
              N2[ss,s] = j1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
              N3[ss,s] = k1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
              N4[ss,s] = l1*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
              
              ss += 1
        s += 1
    
  return array([N1, N2, N3, N4])



def representation(ket0):
  '''
    

    Parameters
    ----------
    ket0 : list/np.array
        Ket |N1, N2, N3, N4> as:
            "[N1, N2, N3, N4]".

    Returns
    -------
    psi : np.array
        Vector representation of the quantum state.

    '''  
        

  M,P,Q,R = ket0
  nb = M+P+Q+R
  dim = int((nb+3)*(nb+2)*(nb+1)/6)      

  s, s0 = 0,0
  
  for i in range(0, nb+1):
    for j in range(0, nb+1-i):
      for k in range(0, nb+1-i-j):
        l = nb - k - j - i
        if (i==M and j==P and k==Q and l==R):
          s0 = s
        
        s += 1
  
  psi = zeros((dim))
  psi[s0] = 1.0
  
  return psi



def make_state(coefs, kets):
  '''
    

    Parameters
    ----------
    coefs : list/np.array
        [c1, c2, ...].
    kets : np.array
        [|Ket1>, |Ket2>, ...].

    Returns
    -------
    psis : np.array
        Representation of state:
            "c1|Ket1> + c2|Ket2> + ...".

    '''  


  from sys import exit

  natom = sum(kets[0])
  dim = int((natom+3)*(natom+2)*(natom+1)/6)
 
  comp_check = any(type(coefs[i]) == np.complex128 for i in range(len(coefs)))
  num_check = any(sum(kets[i]) != natom for i in range(len(kets)))

  if comp_check == True:
    var_type = complex
  else:
    var_type = float

  if num_check == True:
    exit("Total # of atoms not conserved!")

  if len(coefs) != len(kets):
    exit("# of states in superp. mismatch!")
  
  psis = zeros((dim), dtype=var_type)

  for i in range(len(coefs)):
    psis += coefs[i]*representation(kets[i])
  
  return psis 



def d(x,y):
  res = 0
  if x==y:
    res = 1
  return res
