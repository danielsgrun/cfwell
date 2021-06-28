from numpy import zeros


def make_h(params, natom):
  
  '''
  Create hamiltonian representation with parameters provided.
    
  Results:
    Hint: interaction hamiltonian;
    Htun: tunneling hamiltonian
    Hb13: 1-3 breaking (\nu) hamiltonian;
    Hb24: 2-4 breaking (\mu) hamiltonian;
    H:    "complete"" hamiltonian.
  '''
    
  from numpy import array  
  from math import sqrt
    
  U, J = (params['interaction'], params['hopping'])
  nu, mu = (params['break13'], params['break24'])
  nb = natom
      
  dim = int((nb+3)*(nb+2)*(nb+1)/6)  
      
  Hint = zeros((dim,dim))
  Htun = zeros((dim,dim))
    
  Hb13 = zeros((dim,dim))
  Hb24 = zeros((dim,dim))
    
  s, ss = 0, 0

  for i1 in range(0, nb+1):
    for j1 in range(0, nb-i1+1):  
      for k1 in range(0, nb-i1-j1+1):
             
        l1 = nb-i1-j1-k1
        ss = 0
            
        for i2 in range(0, nb+1):
          for j2 in range(0, nb-i2+1):
            for k2 in range(0, nb-i2-j2+1):
              l2 = nb-i2-j2-k2
                  
              Hint[ss,s] = ((i1+k1-j1-l1)**2 
                                 *d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1) )
                
              Hb13[ss,s] = (i1 - k1)*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)
              Hb24[ss,s] = (j1 - l1)*d(i2,i1)*d(j2,j1)*d(k2,k1)*d(l2,l1)

              Htun[ss,s] = ( (sqrt(i1*(j1+1.))*d(i2,i1-1)*d(j2,j1+1) 
                                     + sqrt(j1*(i1+1.))*d(i2,i1+1)*d(j2,j1-1))
                                     *d(k2,k1)*d(l2,l1)
                                   +(sqrt(j1*(k1+1.))*d(j2,j1-1)*d(k2,k1+1) 
                                     + sqrt(k1*(j1+1.))*d(j2,j1+1)*d(k2,k1-1))
                                     *d(i2,i1)*d(l2,l1)
                                   +(sqrt(k1*(l1+1.))*d(k2,k1-1)*d(l2,l1+1) 
                                     + sqrt(l1*(k1+1.))*d(k2,k1+1)*d(l2,l1-1))
                                     *d(i2,i1)*d(j2,j1)
                                   +(sqrt(i1*(l1+1.))*d(i2,i1-1)*d(l2,l1+1) 
                                     + sqrt(l1*(i1+1.))*d(i2,i1+1)*d(l2,l1-1))
                                     *d(j2,j1)*d(k2,k1) )
              ss += 1
        s += 1
                  
    
  H = (-U * Hint - 0.5*J * Htun + 
              nu * Hb13 + mu * Hb24)

  return array([Hint, Htun, Hb13, Hb24, H])
        
        
    
def bands(H_struc, params, natom, plot=True, num=500):
  '''
    

    Parameters
    ----------
    H_struc : numpy.array
        Contains the structure of: [Hint, Htun, Hbreak13, Hbreak24]
    params : dictionary
        Dictionary containing: {'interaction': U
                                'hopping': J,
                                'break13': nu,
                                'break24': mu}.
    natom : int
        Total # of atoms.
    plot : bool, optional
        Decide whether to plot or not. The default is True.
    num : int, optional
        # of plotting points. The default is 500.

    Returns
    -------
    np.array([U/J, bvals]).

    '''    
  
  from numpy import linspace, shape, array
  from scipy.linalg import eigh, eigvals
  import matplotlib.pyplot as plt
  from sys import exit
  from tqdm import tqdm

  Hint, Htun = H_struc[0], H_struc[1]
  Hb13, Hb24 = H_struc[2], H_struc[3]

  U, J = (params['interaction'], params['hopping'])
  nu, mu = (params['break13'], params['break24'])
  nb = natom
      
  dim = int((nb+3)*(nb+2)*(nb+1)/6)

  if dim != shape(Hint)[0]:
    print("Dimension mismatch!")
    exit
      
  Urange = linspace(0, 6 * J, num=500)
  bvals = zeros((len(Urange), dim))
      
 
  bvals[0] = eigvals(-Urange[0]*Hint - 0.5*J*Htun)  
  bvals[0].sort() 
  
    
  for i in tqdm(range(1,len(Urange))):
   
    bvals[i] = eigvals(-Urange[i]*Hint - 0.5*J*Htun
                         + mu*Hb24 + nu*Hb13)
    bvals[i].sort()     
         
  if plot==True:
      
    emax = bvals[0][-1]/J
    plt.figure(figsize=(11,8))
      
    for i in range(0, dim):
          
      plt.plot(Urange/J, bvals[:,i]/J,alpha=0.4,lw=1.0, color='cyan')
      plt.xlabel("$U/J$", fontsize=21)
      plt.xticks(fontsize=21)
      plt.yticks(fontsize=21)
      plt.xlim(0,Urange[-1]/J)
      plt.xlim(0,1.5)
      plt.ylim(-3 * emax, emax)
      plt.ylabel("E/J", fontsize=21)
      plt.axvline(x=U/J, color='k', ls='--', alpha=0.7)      
      
    
    plt.show()  

  return array([Urange/J, bvals])


def d(x,y):
  res = 0
  if x==y:
    res = 1
  return res