import numpy as np 

π = np.pi 

def Expm(A):
    Eigenvalues, Eigenvectors = np.linalg.eig( A )
    return Eigenvectors @  np.diag(  np.exp( Eigenvalues)  )  @ np.linalg.inv(Eigenvectors)

def getAO(DA, DB, Ca, Cb):
    ## transform MO density into an AO density 
    return ( (Ca)@ DA @(Ca.T) ), ( (Cb)@ DB @(Cb.T) )

def getMO(FA, FB, Ca, Cb):
    ## transform AO Fock into an MO Fock 
    return ( (Ca.T)@ FA @(Ca) ), ( (Cb.T)@ FB @(Cb) )

def getFMX(Da, Db, H1, H2):
    J  = np.sum(H2*(Da+Db), axis=(2,3))
    Fα = H1 + J - np.sum(H2.swapaxes(1,2)*Da, axis=(2,3))
    Fβ = H1 + J - np.sum(H2.swapaxes(1,2)*Db, axis=(2,3))
    return Fα, Fβ

def RTHF_setup(uhf_pyscf, D=None, dt=0.002, dT=100, field=None, current=False):
    """
    GIVEN:  dt (electronic time-step), 
            dT (nuclei time-step), 
            D (initial MO density-matrix 3D numpy D_spq ),
            uhf_pyscf (uhf class)
            field (2D numpy array representing real-time electric-field)
    GET:    d_tx: dynamic dipole
            energy: energy of system in time
            trace: trace of MO density matrix in time
            Dao: AO current
    """

    tsteps = int(dT/dt)
    t      = np.arange(0, dT, dt)
    if field is None:
        field = np.zeros((tsteps,3))

    if D is None:
        DA_mo, DB_mo = (uhf_pyscf).mo_occ
        DA_mo  = np.diag(DA_mo).astype(complex)
        DB_mo  = np.diag(DB_mo).astype(complex)
    else:
        DA_mo, DB_mo = D[0], D[1]

    Ca, Cb = (uhf_pyscf).mo_coeff
    dipole = (uhf_pyscf.mol).intor("int1e_r")
    H1     = (uhf_pyscf.mol).intor("int1e_kin")
    H1    += (uhf_pyscf.mol).intor("int1e_nuc")
    H2     = (uhf_pyscf.mol).intor("int2e") ###!! dont calculate!!

    if current:
        D_tspq, Dao_out, d_tx, trace, energy = RTHF(Ca, Cb, DA_mo, DB_mo, H1, H2, field, dipole, dt, current=True)
        return t, d_tx, energy, trace, D_tspq, Dao_out
    else:
        Dao_out, d_tx, trace, energy = RTHF(Ca, Cb, DA_mo, DB_mo, H1, H2, field, dipole, dt, current=False)
        return t, d_tx, energy, trace, Dao_out

def RTHF(Ca, Cb, DA_mo, DB_mo, H1, H2, E_t, dipole, dt, current=False):

    tsteps = len(E_t)
    d_tx   = np.zeros((tsteps, 3))
    energy = np.zeros(tsteps)
    trace  = np.zeros(tsteps)
    Davg   = np.zeros((2, len(Ca), len(Ca)), dtype=np.complex128)
    if current:
        D_out  = np.zeros((tsteps, 2, len(Ca), len(Ca)), dtype=np.complex128)
    for step in (range(tsteps)):
        t = (step) * dt
        
        DA_ao, DB_ao = getAO(DA_mo, DB_mo, Ca, Cb)
        Fa_ao, Fb_ao = getFMX(DA_ao, DB_ao, H1, H2) - dipole[0]*E_t[step,0] - dipole[1]*E_t[step,1] - dipole[2]*E_t[step,2]
        FA_mo, FB_mo = getMO(Fa_ao, Fb_ao, Ca, Cb)

        #### probe
        DD           = ( DA_ao + DB_ao ).real
        #D_out[step]  = np.array([DA_ao, DB_ao])
        Davg += np.array([DA_ao, DB_ao])
        d_tx[step]   = np.array([np.trace(dipole[0] @ DD), np.trace(dipole[1] @ DD), np.trace(dipole[2] @ DD)])
        trace[step]  = np.trace(DA_mo.real + DA_mo.real)
        energy[step] = np.trace((H1 + Fa_ao) @ DA_ao/2).real + np.trace((H1 + Fb_ao) @ (DB_ao/2)).real
        if current:
            D_out[step]  = np.array([DA_mo, DB_mo])

        #### unitary propagators
        UA  = Expm( -1j*(dt)*FA_mo )
        UB  = Expm( -1j*(dt)*FB_mo )

        DA_mo = (UA) @ DA_mo @ ((UA).conj().T)
        DB_mo = (UB) @ DB_mo @ ((UB).conj().T)

    d_tx -= np.einsum("tx -> x", d_tx)/len(d_tx)
    if current:
        return D_out, Davg, d_tx, trace, energy
    else:
        return Davg, d_tx, trace, energy


class E_field(object):
    ''' A class of Electric Field Pulses '''
    def __init__(self, vector=[0.0, 0.0, 1.0], ω=0., E0=1., Γ=np.inf, t0=0., phase=0.):
        self.vector = np.asarray(vector)
        self.ω = ω
        self.Γ = Γ
        self.phase = phase
        self.t0 = t0
        self.E0 = E0
        
        self.timeline  = None
        self.E_t       = None
        self.freqspace = None
        self.E_ω       = None
        
    def getE(self, t, get_Real=True):
        """ Scalar E-field for a given time/instant t """
        self.E_t = self.E0 * np.exp( - 4*np.log(2) * (t - self.t0)**2/(self.Γ**2) ) * np.exp( -1j * self.ω * (t - self.t0) + 1j * self.phase )
        if get_Real:
            return self.E_t.real
        else:
            return self.E_t
    
    def getEE(self, t):
        """ Vectored E-field for a given time/instant t """
        E_t = ( self.E0 * np.exp( - 4*np.log(2) * (t - self.t0)**2/(self.Γ**2) ) * np.sin( self.ω * (t - self.t0) + 1j * self.phase ))
        return np.array([ E_t * (self.vector[0]), E_t * (self.vector[1]), E_t * (self.vector[2])] ).T
        #return ( self.E0 * np.exp( - 4*np.log(2) * (t - self.t0)**2/(self.Γ**2) ) * np.exp( -1j * self.ω * (t - self.t0) + 1j * self.phase )) * (self.vector)
    
    def getEE_whole(self, t):
        """ Vectored E-field for a given time/instant t """
        t = np.asarray([t])
        return np.einsum("t, x -> tx", self.E0 * np.exp( - 4*np.log(2) * (t - self.t0)**2/(self.Γ**2) ) * np.exp( -1j * self.ω * (t - self.t0) + 1j * self.phase ), (self.vector) )
    
    def getEω(self, ω):
        """ Analytic Fourier Transform for Gaussian Wavepacket """
        self.E_ω = self.E0 * self.Γ / (np.sqrt( 8*np.log(2) )) * np.exp( - ( self.Γ**2 * (self.ω - ω)**2 )/( 16*np.log(2) ) ) * np.exp( 1j * (self.phase + ω * self.t0) )
        return self.E_ω
    
    def get_freq(self, timeline, AngularFreq=True):
        """ given timeline (time-array) get corresponding frequency array for FFT """
        T  = timeline[-1]
        dt = timeline[1] - timeline[0]
        f  = np.arange(0.,1/dt + 1/T, 1/T)[:int(len(timeline)/2)]
        
        if AngularFreq:
            self.freqspace = 2*np.pi*f
            return 2*np.pi*f
        else:
            self.freqspace = f
            return f

    def FFT_1D(self, A_t):
        return (np.fft.fft(  A_t  ))[:int(len(A_t)/2)] / np.pi
    
    def get_all(self, t):
        self.getE(t)
        self.get_freq(t)
        self.getEω(self.freqspace)
        return None

