#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:34:14 2021

ε ρ τ υ θ ι ο π α σ δ φ γ η ξ κ λ ζ χ ψ ω β ν μ 
Ε Ρ Τ Υ Θ Ι Ο Π Α Σ Δ Φ Γ Η Ξ Κ Λ Ζ Χ Ψ Ω Β Ν Μ
ς 
Å, å, ð, æ, ø, 
ä, ë, ï, ö, ü,

@author: julio
"""

# =============================================================================
# UHF Class
# =============================================================================

import numpy as np
from pyscf import scf, gto
import scipy.linalg as linalg
from tqdm import tqdm

Z_mass = np.array([ 1.,   1837.,   7297.,  
                   12650., 16427.,  19705.,  21894.,  25533.,  29164.,  34631., 36785.,  
                   41908., 44305.,  49185.,  51195.,  56462.,  58441.,  64621.,  72820.,  
                   71271., 73057.,  81949.,  87256.,  92861.,  94782., 100145., 101799., 107428., 106990., 115837., 119180., 127097., 132396., 136574., 143955., 145656., 152754.,
                   155798., 159721., 162065., 166291., 169357., 174906., 176820., 184239., 187586., 193991., 196631., 204918., 209300., 216395., 221954., 232600., 231331., 239332., 
                   242270., 250331., 253208., 255415., 256859., 262937., 264318., 274089., 277013., 286649., 289702., 296219., 300649., 304894., 307947., 315441., 318945., 325367., 329848., 335119., 339434., 346768., 350390., 355616., 359048., 365656., 372561., 377702., 380947., 380983., 382806., 404681.,
                   406504., 411972., 413795., 422979., 421152., 433900., 432024., 444784., 442961., 450253., 450253., 457545., 459367., 468482., 470305., 472128., 477596., 486711., 492179., 490357., 492179., 492179., 506763., 512231., 512231., 519523., 521346., 526814., 526814., 534106., 534106., 535929.])

Z_dictonary = np.array(['e ', 'H',  'He', 
                        'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',
                         'K', 'Ca', 'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr',  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',  'I', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',  'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa',  'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])

Element_Names = { 'e':0,  0:'e' ,   'H':  1,  1:'H' ,  'He':  2,  2:'He',
               'Li':  3,  3:'Li',  'Be':  4,  4:'Be',   'B':  5,  5:'B' ,   'C':  6,  6:'C' ,   'N':  7,  7:'N' ,   'O':  8,  8:'O' ,   'F':  9,  9:'F' ,  'Ne': 10, 10:'Ne', 
               'Na': 11, 11:'Na',  'Mg': 12, 12:'Mg',  'Al': 13, 13:'Al',  'Si': 14, 14:'Si',   'P': 15, 15:'P' ,   'S': 16, 16:'S' ,  'Cl': 17, 17:'Cl',  'Ar': 18, 18:'Ar',
                'K': 19, 19:'K' ,  'Ca': 20, 20:'Ca',  'Sc': 21, 21:'Sc',  'Ti': 22, 22:'Ti',   'V': 23, 23:'V' ,  'Cr': 24, 24:'Cr',  'Mn': 25, 25:'Mn',  'Fe': 26, 26:'Fe',  'Co': 27, 27:'Co',  'Ni': 28, 28:'Ni',  'Cu': 29, 29:'Cu',  'Zn': 30, 30:'Zn',  'Ga': 31, 31:'Ga',  'Ge': 32, 32:'Ge',  'As': 33, 33:'As',  'Se': 34, 34:'Se',  'Br': 35, 35:'Br',  'Kr': 36, 36:'Kr', 
               'Rb': 37, 37:'Rb',  'Sr': 38, 38:'Sr',   'Y': 39, 39:'Y' ,  'Zr': 40, 40:'Zr',  'Nb': 41, 41:'Nb',  'Mo': 42, 42:'Mo',  'Tc': 43, 43:'Tc',  'Ru': 44, 44:'Ru',  'Rh': 45, 45:'Rh',  'Pd': 46, 46:'Pd',  'Ag': 47, 47:'Ag',  'Cd': 48, 48:'Cd',  'In': 49, 49:'In',  'Sn': 50, 50:'Sn',  'Sb': 51, 51:'Sb',  'Te': 52, 52:'Te',   'I': 53, 53:'I' ,  'Xe': 54, 54:'Xe',
               'Cs': 55, 55:'Cs',  'Ba': 56, 56:'Ba',  'La': 57, 57:'La',  'Ce': 58, 58:'Ce',  'Pr': 59, 59:'Pr',  'Nd': 60, 60:'Nd',  'Pm': 61, 61:'Pm',  'Sm': 62, 62:'Sm',  'Eu': 63, 63:'Eu',  'Gd': 64, 64:'Gd',  'Tb': 65, 65:'Tb',  'Dy': 66, 66:'Dy',  'Ho': 67, 67:'Ho',  'Er': 68, 68:'Er',  'Tm': 69, 69:'Tm',  'Yb': 70, 70:'Yb',  'Lu': 71, 71:'Lu',  'Hf': 72, 72:'Hf',  'Ta': 73, 73:'Ta',   'W': 74, 74:'W' ,  'Re': 75, 75:'Re',  'Os': 76, 76:'Os',  'Ir': 77, 77:'Ir',  'Pt': 78, 78:'Pt',  'Au': 79, 79:'Au', 'Hg': 80, 80:'Hg',  'Tl': 81, 81:'Tl',  'Pb': 82, 82:'Pb',  'Bi': 83, 83:'Bi',  'Po': 84, 84:'Po', 'At': 85, 85:'At', 'Rn': 86, 86:'Rn', 
               'Fr': 87, 87:'Fr',  'Ra': 88, 88:'Ra',  'Ac': 89, 89:'Ac',  'Th': 90, 90:'Th',  'Pa': 91, 91:'Pa',   'U': 92, 92:'U' ,  'Np': 93, 93:'Np',  'Pu': 94, 94:'Pu',  'Am': 95, 95:'Am',  'Cm': 96, 96:'Cm',  'Bk': 97, 97:'Bk',  'Cf': 98, 98:'Cf',  'Es': 99, 99:'Es',  'Fm':100,100:'Fm',  'Md':101,101:'Md',  'No':102,102:'No',  'Lr':103,103:'Lr',  'Rf':104,104:'Rf',  'Db':105,105:'Db',  'Sg':106,106:'Sg',  'Bh':107,107:'Bh',  'Hs':108,108:'Hs',  'Mt':109,109:'Mt',  'Ds':110,110:'Ds',  'Rg':111,111:'Rg', 'Cn':112,112:'Cn',  'Nh':113,113:'Nh',  'Fl':114,114:'Fl',  'Mc':115,115:'Mc',  'Lv':116,116:'Lv', 'Ts':117,117:'Ts', 'Og':118,118:'Og'}


def Bohr(x): ### given Ångström get Bohr
    return 1.889726125 * x 

def Ångström(x): ### given Bohr get Ångström
    return x / 1.889726125

def print2darray_tostring(Array_2D, ZZ):
    """ ZZ is the input from UHF object (LiH.Z) """
    output = ''
    for index, atom in enumerate(Array_2D):
        output += "  " + ZZ[index]
        for xyz in atom:
            number = np.format_float_positional(np.float32(xyz), unique=False, precision=6, pad_left = True)
            output += "\t  " + number
        output += ";\n"
    return output.rstrip()

def Get_XYZ_XYZ_t(NumPy_Array, Z_in, file_name):
    #reorinate the array, i = atom #, x = xyz coordinate, t = time-step
    #NumPy_Array = np.einsum('ixt -> tix', NumPy_Array) 
    
    #http://www.chm.bris.ac.uk/~paulmay/temp/pcc/xyz.htm
    ff = open(file_name + ".xyz", "w")
    for index, element in enumerate(NumPy_Array):
        ff.write(str(len(element)) + "\n")
        ff.write(str(index) + "\n")
        ff.write(print2darray_tostring(element, Z_in))
        ff.write("\n")
    ff.close() 
    
    return None

def h_deriv(atom_id, h1, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]
    with mol.with_rinv_at_nucleus(atom_id):
        vrinv  = (-mol.atom_charge(atom_id)) * mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
        vrinv += mol.intor('ECPscalar_iprinv', comp=3)
    vrinv[:,p0:p1] += h1[:,p0:p1]
    return vrinv + vrinv.swapaxes(1,2) 
def S_deriv(atom_id, S_xAB, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]

    vrinv = np.zeros(S_xAB.shape)
    vrinv[:, p0:p1, :] += S_xAB[:, p0:p1, :]
    
    return vrinv + vrinv.swapaxes(1,2)
def I_deriv(atom_id, I_xABCD, mol):
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]

    vrinv  = np.zeros(I_xABCD.shape)
    vrinv[:, p0:p1, :, :, :] += I_xABCD[:, p0:p1, :, :, :]
    
    vrinv += np.einsum("xABCD -> xCDAB", vrinv) 
    vrinv += np.einsum("xABCD -> xBACD", vrinv) 
    vrinv += np.einsum("xABCD -> xABDC", vrinv)
    vrinv += np.einsum("xABCD -> xBADC", vrinv)
    
    return vrinv/4.

def grad_nuc(mol, atmlst=None):
    '''
    Author: Qiming Sun <osirpt.sun@gmail.com>
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    gs = np.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.sqrt(np.dot(r1-r2,r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs

def RT_FFT(t, A_tx, Γ=0.01):
    dt   = t[2] - t[1]
    ω    = 2*np.pi * np.linspace(0.0, 1.0/(2.0*dt), int(len(t)/2))
    #d_ωx = 2.0/len(t) * (scipy.fftpack.fft( A_tx * 1.0 , axis=0))[:len(t)//2, :]
    
    d_ωx = (dt * 2/np.sqrt(2*np.pi) ) * (np.fft.fft( np.einsum("tx, t -> tx", A_tx, np.exp(- Γ * t)) , axis=0))[:int(len(t)/2), :]
    return ω, d_ωx

def Expm(A):
    Eigenvalues, Eigenvectors = np.linalg.eig( A ) ### !!! eigh
    return Eigenvectors @  np.diag(  np.exp( Eigenvalues)  )  @ np.linalg.inv(Eigenvectors)

class pyscf_UHF(object):
    # 6/30/21
    ''' A class of Hartree-Fock Characters '''
    def __init__(self, Z=None, xyz=None, basis=None):

        ### basics
        self.Z = Z
        self.xyz = xyz
        self.basis = None
        self.mass = None
        self.atomic_num = None
        self.nbf = None
        self.pyscf_mol = None
        
        ### HF Calculation
        self.E = None
        self.Ca = None
        self.Cb = None
        self.Da = None
        self.Db = None
        self.Fa = None
        self.Fb = None
        self.Ea = None
        self.Eb = None
        self.Na = None
        self.Nb = None
        self.Da_mo = None
        self.Db_mo = None
        self.OVa = None
        self.OVb = None
        
        ### integrals
        self.pmints = None
        self.A = None
        self.D = None
        self.I = None
        self.S = None
        self.H = None
        
    def initialize(self, xyz=None, Z=None, charge=0, basis=None, spin=0):
        # 9/30/21
        if xyz is not None:
            if spin == 0:
                try:
                    self.pyscf_mol = gto.M(atom=print2darray_tostring(xyz, Z), charge=charge)
                except:
                    self.pyscf_mol = gto.M(atom=print2darray_tostring(xyz, Z), charge=charge, spin=1)
                
            else:
                print("WENT ELSE")
                self.pyscf_mol = gto.M(atom=print2darray_tostring(xyz, Z), charge=charge, spin=spin)

        if isinstance(basis, str):
            (self.pyscf_mol).basis = basis
        
        (self.pyscf_mol).verbose = 1 ### why not 0??
        charges = (self.pyscf_mol).atom_charges()
        CoM = np.einsum('i, ix-> x', (self.pyscf_mol).atom_mass_list(isotope_avg=True), (self.pyscf_mol).atom_coords() ) / np.sum( (self.pyscf_mol).atom_mass_list(isotope_avg=True) )
        #CoM = np.einsum('i, ix-> x', mole.atom_mass_list(isotope_avg=False), mole.atom_coords() ) / np.sum( mole.atom_mass_list(isotope_avg=False) )
        (self.pyscf_mol).set_common_orig_( CoM ) ### ohh partial charges??? 
        
        (self.pyscf_mol).set_common_orig_(CoM)
        (self.pyscf_mol).build()
        
        self.mass = ((self.pyscf_mol()).atom_mass_list())*1822.89
        self.xyz  = (self.pyscf_mol()).atom_coords()
        self.Z    = np.asarray([element[0] for element in (self.pyscf_mol())._atom])
        
    def Calc(self):
        UHF_pyscf = scf.UHF(self.pyscf_mol) #, verbose=0)
        self.pyscfuhf = UHF_pyscf
        (self.pyscfuhf).conv_tol = 1e-12
        (self.pyscfuhf).kernel()
    
        self.Ca, self.Cb = (self.pyscfuhf).mo_coeff
        self.Da, self.Db = (self.pyscfuhf).make_rdm1()
        self.Ea, self.Eb = (self.pyscfuhf).mo_energy
        self.Oa, self.Ob = (self.pyscfuhf).mo_occ
        self.E = (self.pyscfuhf).e_tot 
        
        self.Da_mo = np.diag(self.Oa).astype(complex)
        self.Db_mo = np.diag(self.Ob).astype(complex)
        self.Na = int(np.sum(self.Oa))
        self.Nb = int(np.sum(self.Ob))
        
        self.Cao = self.Ca[:, :self.Na]
        self.Cav = self.Ca[:, self.Na:]
        self.Cbo = self.Cb[:, :self.Nb]
        self.Cbv = self.Cb[:, self.Nb:]
        self.Eao = self.Ea[:self.Na]
        self.Eav = self.Ea[self.Na:]
        self.Ebo = self.Eb[:self.Nb]
        self.Ebv = self.Eb[self.Nb:]
        
        self.D = self.get_dipole_ints() #(self.pyscf_mol).intor('int1e_r')
        self.I = (self.pyscf_mol).intor("int2e") # self.pyscfuhf._eri #(self.pyscf_mol).intor("int2e")
        self.S = (self.pyscf_mol).intor("int1e_ovlp")   
        self.H = (self.pyscf_mol).intor('int1e_kin') + (self.pyscf_mol).intor('int1e_nuc') + (self.pyscf_mol).intor("ECPscalar")
        
        eig, v = np.linalg.eigh(self.S)
        self.A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)
        
        self.OVa = (self.Na) * ( len(self.S) - (self.Na) )
        self.OVb = (self.Nb) * ( len(self.S) - (self.Nb) )
        
        self.Fa, self.Fb = self.get_UFock(self.Da, self.Db)
        self.nuclei_energy = (self.pyscfuhf).energy_nuc()
        #self.nuclei_force  = grad_nuc(self.pyscfuhf)
        
        ### self energies
        J = np.einsum('ABCD, CD -> AB', self.I, self.Da + self.Db , optimize=True)
        Kα = np.einsum('ABCD, BC -> AD', self.I, self.Da , optimize=True)
        Kβ = np.einsum('ABCD, BC -> AD', self.I, self.Db , optimize=True)
        self.Σα = J - Kα
        self.Σβ = J - Kβ
        
        return None
    
    def get_dipole_ints(self):
        mol = self.pyscf_mol
        return -mol.intor('cint1e_r_sph', comp=3)
        
    def get_UFock(self, Da, Db):
        J  = np.einsum('ABCD, CD -> AB', self.I, Da + Db , optimize=True)
        Kα = np.einsum('ABCD, BC -> AD', self.I, Da , optimize=True)
        Kβ = np.einsum('ABCD, BC -> AD', self.I, Db , optimize=True)
        Fα = self.H + J - Kα
        Fβ = self.H + J - Kβ
        return Fα,  Fβ
    
    def get_UCFock(self, Da, Db):
        J  = np.einsum('ABCD, CD -> AB', (self.I).astype(complex), Da + Db , optimize=True)
        Kα = np.einsum('ABCD, BC -> AD', (self.I).astype(complex), Da , optimize=True)
        Kβ = np.einsum('ABCD, BC -> AD', (self.I).astype(complex), Db , optimize=True)
        Fα = (self.H).astype(complex) + J - Kα
        Fβ = (self.H).astype(complex) + J - Kβ
        return Fα,  Fβ
    
    def get_UCFockSS(self, Da, Db):
        #J  = np.einsum('ABCD, CD -> AB', (self.I).astype(complex), self.Da + self.Db , optimize=True)
        #Kα = np.einsum('ABCD, BC -> AD', (self.I).astype(complex), self.Da , optimize=True)
        #Kβ = np.einsum('ABCD, BC -> AD', (self.I).astype(complex), self.Db , optimize=True)
        Fα = (self.H).astype(complex) + self.Σα
        Fβ = (self.H).astype(complex) + self.Σβ
        return Fα,  Fβ
    
    def get_Euhf(self, DA, DB, FA, FB):
        Euhf  = np.einsum("pq,pq->", DA + DB, self.H)
        Euhf += np.einsum("pq,pq->", DA, FA)
        Euhf += np.einsum("pq,pq->", DB, FB)
        Euhf *= 0.5
        return Euhf
    
    def dints(self):
        mol = self.pyscf_mol
        h_xAB    = -mol.intor('ECPscalar_ipnuc', comp=3)
        h_xAB   += -mol.intor('int1e_ipkin', comp=3)
        h_xAB   += -mol.intor('int1e_ipnuc', comp=3)
        S_xAB    = -mol.intor('int1e_ipovlp', comp=3)
        I_xABCD  = -mol.intor('int2e_ip1', comp=3)
        dI_pyscf = np.zeros(((len(mol.aoslice_by_atom()),) + I_xABCD.shape))
        dH_pyscf = np.zeros(((len(mol.aoslice_by_atom()),) + h_xAB.shape))
        dS_pyscf = np.zeros(((len(mol.aoslice_by_atom()),) + S_xAB.shape))
        for i in range(len(mol.aoslice_by_atom())):
            dI_pyscf[i] = I_deriv(i, I_xABCD, mol)
            dS_pyscf[i] = S_deriv(i, S_xAB, mol)
            dH_pyscf[i] = h_deriv(i, h_xAB, mol)
            
        self.dI = dI_pyscf
        self.dS = dS_pyscf
        self.dH = dH_pyscf
        return None
    
    def get_force(self, DA, DB, FA, FB): ## added may 22
        D = DA + DB    
    
        ## Hellman Feynman
        fix  = -grad_nuc(self.pyscf_mol)
        fix -= 1.0*np.einsum('mn, IXmn -> IX',  D, self.dH)
        fix -= 0.5*np.einsum('nm, ls, IXmnls -> IX',  D,  D, self.dI, optimize=True) ## dJ
        fix += 0.5*np.einsum('nm, ls, IXmlsn -> IX', DA, DA, self.dI, optimize=True) ## dKα
        fix += 0.5*np.einsum('nm, ls, IXmlsn -> IX', DB, DB, self.dI, optimize=True) ## dKβ
        
        ## Pulay
        fix += 1.0*np.einsum('ij, IXjk, kl, il -> IX', DA, self.dS, DA, FA, optimize=True)
        fix += 1.0*np.einsum('ij, IXjk, kl, il -> IX', DB, self.dS, DB, FB, optimize=True)
        return fix
    
    def get_Cforce(self, DA, DB, FA, FB): ## added may 22
        D = DA + DB    
    
        ## Hellman Feynman
        fix  = -grad_nuc(self.pyscf_mol).astype(complex)
        fix -= 1.0*np.einsum('mn, IXmn -> IX',  D, (self.dH).astype(complex))
        fix -= 0.5*np.einsum('nm, ls, IXmnls -> IX',  D,  D, (self.dI).astype(complex), optimize=True) ## dJ
        fix += 0.5*np.einsum('nm, ls, IXmlsn -> IX', DA, DA, (self.dI).astype(complex), optimize=True) ## dKα
        fix += 0.5*np.einsum('nm, ls, IXmlsn -> IX', DB, DB, (self.dI).astype(complex), optimize=True) ## dKβ
        
        ## Pulay
        fix += 1.0*np.einsum('ij, IXjk, kl, il -> IX', DA, (self.dS).astype(complex), DA, FA, optimize=True)
        fix += 1.0*np.einsum('ij, IXjk, kl, il -> IX', DB, (self.dS).astype(complex), DB, FB, optimize=True)
        return fix
    
    def get_dM(self, return_AO=False):
        
        dΠ = self.dI - np.einsum("ixABCD -> ixADCB", self.dI)
        Π  = self.I - np.einsum("ABCD -> ADCB", self.I)
        
        QA = -1.*np.einsum('ixAB, Ap, Cp -> ixBC', self.dS, self.Ca, self.Ca, optimize=True)/2
        QB = -1.*np.einsum('ixAB, Ap, Cp -> ixBC', self.dS, self.Cb, self.Cb, optimize=True)/2
        DA_ix = 1.*np.einsum('ixNA, NB -> ixAB', QA, self.Da)
        DB_ix = 1.*np.einsum('ixNA, NB -> ixAB', QB, self.Db)
        
        MA_ao  = -0.5*self.dH
        MA_ao -=  0.5*np.einsum('db, AXagbd -> AXga', self.Da, dΠ) 
        MA_ao -=  0.5*np.einsum('db, AXagbd -> AXga', self.Db, self.dI)
        MA_ao +=  0.5*np.einsum('ixAB, Mp, Ap, MN -> ixNB', self.dS, self.Ca, self.Ca, self.Fa, optimize=True)
        MA_ao += -1.*np.einsum("ABCD, ixCD -> ixAB", Π, DA_ix)
        MA_ao += -1.*np.einsum("ABCD, ixCD -> ixAB", self.I, DB_ix)
        MA_ao  = MA_ao + MA_ao.swapaxes(2,3)
        
        MB_ao  = -0.5*self.dH
        MB_ao -=  0.5*np.einsum('db, AXagbd -> AXga', self.Db, dΠ)
        MB_ao -=  0.5*np.einsum('db, AXagbd -> AXga', self.Da, self.dI)
        MB_ao +=  0.5*np.einsum('ixAB, Mp, Ap, MN -> ixNB', self.dS, self.Cb, self.Cb, self.Fb, optimize=True)
        MB_ao += -1.*np.einsum("ABCD, ixCD -> ixAB", Π, DB_ix) 
        MB_ao += -1.*np.einsum("ABCD, ixCD -> ixAB", self.I, DA_ix)
        MB_ao  = MB_ao + MB_ao.swapaxes(2,3)
        
        MA_mo = np.einsum("Aj, ixAB, Bb -> ixbj", self.Cao, MA_ao, self.Cav)
        MB_mo = np.einsum("Aj, ixAB, Bb -> ixbj", self.Cbo, MB_ao, self.Cbv)
        MA_I  = (MA_mo).reshape( (MA_mo.shape[0], MA_mo.shape[1], self.OVa) , order="C")
        MB_I  = (MB_mo).reshape( (MB_mo.shape[0], MB_mo.shape[1], self.OVb) , order="C")
        
        self.M_A = MA_I
        self.M_B = MB_I
        
        if return_AO:
            return MA_ao, MB_ao
        else:
            return None
        
    def splitX(self, returnI=False):
        
        XA  = 1.*self.CIS_X[:,:self.OVa]
        XAA = 1.*self.CIS_X[:,self.OVa:]
        
        XA_Iia = np.reshape( XA, ( XA.shape[0], self.Na, len(self.S)-self.Na), order="C")
        XB_Jjb = np.reshape(XAA, (XAA.shape[0], self.Nb, len(self.S)-self.Nb), order="C")
        
        if returnI==False:
            return XA_Iia, XB_Jjb
        else:
            return XA, XAA
    
    def get_Lagrangian(self, XA_ai, XB_ai, return_proto=False):
        
        Π  = self.I - np.einsum("ABCD -> ADCB", self.I)
        
        XA_ao_pieces = np.einsum("Ai, Iia, Ba -> IAB", self.Cao, XA_ai, self.Cav, optimize=True)
        XB_ao_pieces = np.einsum("Ai, Iia, Ba -> IAB", self.Cbo, XB_ai, self.Cbv, optimize=True)
        
        pLA_aaIJ  = -1.*np.einsum('Ba, IDC, ABCD, Jia -> IJAi', self.Cav, XA_ao_pieces, Π, XA_ai, optimize=True)
        pLA_aaIJ += -1.*np.einsum('Ba, IDC, ABCD, Jia -> IJAi', self.Cav, XB_ao_pieces, self.I, XA_ai, optimize=True)
        pLB_aaIJ  = -1.*np.einsum('Ba, IDC, ABCD, Jia -> IJAi', self.Cbv, XB_ao_pieces, Π, XB_ai, optimize=True)
        pLB_aaIJ += -1.*np.einsum('Ba, IDC, ABCD, Jia -> IJAi', self.Cbv, XA_ao_pieces, self.I, XB_ai, optimize=True)
        
        pLA_aaJI  = -1.*np.einsum('Ba, JDC, ABCD, Iia -> IJAi', self.Cav, XA_ao_pieces, Π, XA_ai, optimize=True)
        pLA_aaJI += -1.*np.einsum('Ba, JDC, ABCD, Iia -> IJAi', self.Cav, XB_ao_pieces, self.I, XA_ai, optimize=True)
        pLB_aaJI  = -1.*np.einsum('Ba, JDC, ABCD, Iia -> IJAi', self.Cbv, XB_ao_pieces, Π, XB_ai, optimize=True)
        pLB_aaJI += -1.*np.einsum('Ba, JDC, ABCD, Iia -> IJAi', self.Cbv, XA_ao_pieces, self.I, XB_ai, optimize=True)
        pLA_aa = pLA_aaIJ + pLA_aaJI
        pLB_aa = pLB_aaIJ + pLB_aaJI
        
        pLA_iiJI  = 1.*np.einsum('Aj, JBC, ADCB, Ija -> IJaD', self.Cao, XA_ao_pieces, Π, XA_ai, optimize=True)
        pLA_iiJI += 1.*np.einsum('Aj, JBC, ADCB, Ija -> IJaD', self.Cao, XB_ao_pieces, self.I, XA_ai, optimize=True)
        pLB_iiJI  = 1.*np.einsum('Aj, JBC, ADCB, Ija -> IJaD', self.Cbo, XB_ao_pieces, Π, XB_ai, optimize=True)
        pLB_iiJI += 1.*np.einsum('Aj, JBC, ADCB, Ija -> IJaD', self.Cbo, XA_ao_pieces, self.I, XB_ai, optimize=True)
        
        pLA_iiIJ  = 1.*np.einsum('Aj, IBC, ADCB, Jja -> IJaD', self.Cao, XA_ao_pieces, Π, XA_ai, optimize=True)
        pLA_iiIJ += 1.*np.einsum('Aj, IBC, ADCB, Jja -> IJaD', self.Cao, XB_ao_pieces, self.I, XA_ai, optimize=True)
        pLB_iiIJ  = 1.*np.einsum('Aj, IBC, ADCB, Jja -> IJaD', self.Cbo, XB_ao_pieces, Π, XB_ai, optimize=True)
        pLB_iiIJ += 1.*np.einsum('Aj, IBC, ADCB, Jja -> IJaD', self.Cbo, XA_ao_pieces, self.I, XB_ai, optimize=True)
        pLA_ii = pLA_iiJI + pLA_iiIJ
        pLB_ii = pLB_iiJI + pLB_iiIJ
        
        pLA_rdm  = 1.*np.einsum('Di, IJBA, ABCD -> IJCi', self.Cao, self.RDM_A, Π, optimize=True)
        pLA_rdm += 1.*np.einsum('Di, IJBA, ABCD -> IJCi', self.Cao, self.RDM_B, self.I, optimize=True)
        pLB_rdm  = 1.*np.einsum('Di, IJBA, ABCD -> IJCi', self.Cbo, self.RDM_B, Π, optimize=True)
        pLB_rdm += 1.*np.einsum('Di, IJBA, ABCD -> IJCi', self.Cbo, self.RDM_A, self.I, optimize=True)
        
        ThridPulayA  = -1.*np.einsum("Bi, IJAi -> IJBA", self.Cao, pLA_aa)
        ThridPulayA +=  1.*np.einsum("Aa, IJaB -> IJAB", self.Cav, pLA_ii)
        
        ThridPulayB  = -1.*np.einsum("Bi, IJAi -> IJBA", self.Cbo, pLB_aa)
        ThridPulayB +=  1.*np.einsum("Aa, IJaB -> IJAB", self.Cbv, pLB_ii) 
        
        self.thridpulayA = ThridPulayA
        self.thridpulayB = ThridPulayB
        
        LA_ii  = 1.*np.einsum("Di, IJaD -> IJai", self.Cao, pLA_ii)
        LA_aa  = 1.*np.einsum("Ab, IJAi -> IJbi", self.Cav, pLA_aa)
        LA_rdm = 1.*np.einsum("Ca, IJCi -> IJai", self.Cav, pLA_rdm)
        LB_ii  = 1.*np.einsum("Di, IJaD -> IJai", self.Cbo, pLB_ii)
        LB_aa  = 1.*np.einsum("Ab, IJAi -> IJbi", self.Cbv, pLB_aa)
        LB_rdm = 1.*np.einsum("Ca, IJCi -> IJai", self.Cbv, pLB_rdm)
        
        LA = LA_rdm + LA_aa + LA_ii
        LB = LB_rdm + LB_aa + LB_ii
        
        UCISs = len(XA_ai)
        LA_I = LA.reshape( ( UCISs, UCISs, self.OVa) , order="C" )
        LB_I = LB.reshape( ( UCISs, UCISs, self.OVb) , order="C" )
        
        L_I  = np.concatenate( (LA_I, LB_I), axis=2 )
        self.L = L_I

        if return_proto:
            return pLA_rdm, pLB_rdm, pLA_aa, pLB_aa, pLA_ii, pLB_ii
        else:
            return None
    
    def get_1RDM(self, XA_ai, XB_ai):
        
        γA  = np.einsum("Ai, Iia, Kja, Bj -> IKAB", self.Cao, XA_ai,  XA_ai, self.Cao, optimize=True) - np.einsum("Aa, Iia, Kib, Bb -> IKAB",  self.Cav, XA_ai,  XA_ai, self.Cav, optimize=True)
        γB  = np.einsum("Ai, Iia, Kja, Bj -> IKAB", self.Cbo, XB_ai,  XB_ai, self.Cbo, optimize=True) - np.einsum("Aa, Iia, Kib, Bb -> IKAB",  self.Cbv, XB_ai,  XB_ai, self.Cbv, optimize=True)
        γA = γA + γA.swapaxes(2,3)
        γB = γB + γB.swapaxes(2,3)
        
        self.RDM_A = γA
        self.RDM_B = γB
        
        return None
    
    def UCIS(self):
        
        # =============================================================================
        # Preliminary blocks
        # =============================================================================
        
        IA_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        IB_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        KA_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        KB_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        
        # =============================================================================
        # A_UCIS Block
        # =============================================================================
        
        A_AAAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], KA_aiΥδ, optimize=True)
        A_AABB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IA_aiΥδ, optimize=True)
        A_BBAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IB_aiΥδ, optimize=True)
        A_BBBB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], KB_aiΥδ, optimize=True)
        
        FF_AA_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Ea[self.Na:]), np.eye( len( self.Ea[:self.Na]) ))
        FF_AA_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Ea[:self.Na]), np.eye( len( self.Ea[self.Na:]) ))
        
        FF_BB_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Eb[self.Nb:]), np.eye( len( self.Eb[:self.Nb]) ))
        FF_BB_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Eb[:self.Nb]), np.eye( len( self.Eb[self.Nb:]) ))
        
        
        A_AA = A_AAAA.swapaxes(0, 1) + FF_AA_mo
        A_AB = A_AABB.swapaxes(0, 1)
        A_BA = A_BBAA.swapaxes(0, 1)
        A_BB = A_BBBB.swapaxes(0, 1) + FF_BB_mo
        
        AF_AA = 1.*A_AA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na])     , order="F") ##!!!
        AF_AB = 1.*A_AB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) , order="F") ##!!!
        AF_BA = 1.*A_BA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) , order="F")
        AF_BB = 1.*A_BB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb])     , order="F")
        
        A_AA = A_AA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) , order="C") ##!!!
        A_AB = A_AB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) , order="C") ##!!!
        A_BA = A_BA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) )
        A_BB = A_BB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) )
        
        self.A_UCIS = np.asarray( np.bmat([[A_AA, A_AB],[A_BA, A_BB]]) )
        
        B_AAAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", self.Ca[:, self.Na:], self.Ca[:, :self.Na], KA_aiΥδ, optimize=True)
        B_AABB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IA_aiΥδ, optimize=True) 
        B_BBAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IB_aiΥδ, optimize=True)
        B_BBBB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], KB_aiΥδ, optimize=True)
        
        B_AAAA = (B_AAAA.swapaxes(0, 1)).swapaxes(2, 3) # iajb
        B_AABB = (B_AABB.swapaxes(0, 1)).swapaxes(2, 3)
        B_BBAA = (B_BBAA.swapaxes(0, 1)).swapaxes(2, 3)
        B_BBBB = (B_BBBB.swapaxes(0, 1)).swapaxes(2, 3)
        
        
        BF_AA = 1.*B_AAAA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na])     , order="F")
        BF_AB = 1.*B_AABB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) , order="F")
        BF_BA = 1.*B_BBAA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) , order="F")
        BF_BB = 1.*B_BBBB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb])     , order="F")
        
        B_AAAA = B_AAAA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) )
        B_AABB = B_AABB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) )
        B_BBAA = B_BBAA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) )
        B_BBBB = B_BBBB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) )
        
        self.B_TDHF = np.asarray( np.bmat([[B_AAAA,B_AABB],[B_BBAA,B_BBBB]]) )
        
        
        
        AF_UCIS = np.asarray( np.bmat([[AF_AA, AF_AB],[AF_BA, AF_BB]]) )
        BF_TDHF = np.asarray( np.bmat([[BF_AA, BF_AB],[BF_BA, BF_BB]]) )
        self.Hessian = ( AF_UCIS + BF_TDHF )
        
        return None
    
    def UCIS_F(self):
        
        # =============================================================================
        # Preliminary blocks
        # =============================================================================
        
        IA_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        IB_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        KA_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        KB_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        
        # =============================================================================
        # A_UCIS Block
        # =============================================================================
        
        A_AAAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], KA_aiΥδ, optimize=True)
        A_AABB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IA_aiΥδ, optimize=True)
        A_BBAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IB_aiΥδ, optimize=True)
        A_BBBB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], KB_aiΥδ, optimize=True)
        
        FF_AA_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Ea[self.Na:]), np.eye( len( self.Ea[:self.Na]) ))
        FF_AA_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Ea[:self.Na]), np.eye( len( self.Ea[self.Na:]) ))
        
        FF_BB_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Eb[self.Nb:]), np.eye( len( self.Eb[:self.Nb]) ))
        FF_BB_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Eb[:self.Nb]), np.eye( len( self.Eb[self.Nb:]) ))
        
        
        A_AA = A_AAAA.swapaxes(0, 1) + FF_AA_mo
        A_AB = A_AABB.swapaxes(0, 1)
        A_BA = A_BBAA.swapaxes(0, 1)
        A_BB = A_BBBB.swapaxes(0, 1) + FF_BB_mo
        
        A_AA = A_AA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na])     , order="F") ##!!!
        A_AB = A_AB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) , order="F") ##!!!
        A_BA = A_BA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) , order="F")
        A_BB = A_BB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb])     , order="F")
        
        self.A_UCIS = np.asarray( np.bmat([[A_AA, A_AB],[A_BA, A_BB]]) )
        
        B_AAAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", self.Ca[:, self.Na:], self.Ca[:, :self.Na], KA_aiΥδ, optimize=True)
        B_AABB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IA_aiΥδ, optimize=True) 
        B_BBAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IB_aiΥδ, optimize=True)
        B_BBBB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], KB_aiΥδ, optimize=True)
        
        B_AAAA = (B_AAAA.swapaxes(0, 1)).swapaxes(2, 3) # iajb
        B_AABB = (B_AABB.swapaxes(0, 1)).swapaxes(2, 3)
        B_BBAA = (B_BBAA.swapaxes(0, 1)).swapaxes(2, 3)
        B_BBBB = (B_BBBB.swapaxes(0, 1)).swapaxes(2, 3)
        
        B_AAAA = B_AAAA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na])     , order="F")
        B_AABB = B_AABB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) , order="F")
        B_BBAA = B_BBAA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) , order="F")
        B_BBBB = B_BBBB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb])     , order="F")
        
        self.B_TDHF = np.asarray( np.bmat([[B_AAAA,B_AABB],[B_BBAA,B_BBBB]]) )
        #self.Hessian = 1.*( self.A_UCIS + self.B_TDHF )
        
        return None
    
    def TESTUCIS(self):
        
        # =============================================================================
        # Preliminary blocks
        # =============================================================================
        
        IA_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        IB_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        KA_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Ca[:, self.Na:], self.Ca[:, :self.Na], self.I, optimize=True)
        KB_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', self.Cb[:, self.Nb:], self.Cb[:, :self.Nb], self.I, optimize=True)
        
        # =============================================================================
        # A_UCIS Block
        # =============================================================================
        
        A_AAAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], KA_aiΥδ, optimize=True)
        A_AABB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IA_aiΥδ, optimize=True)
        A_BBAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Ca[:, :self.Na], self.Ca[:, self.Na:], IB_aiΥδ, optimize=True)
        A_BBBB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], KB_aiΥδ, optimize=True)
        
        FF_AA_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Ea[self.Na:]), np.eye( len( self.Ea[:self.Na]) ))
        FF_AA_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Ea[:self.Na]), np.eye( len( self.Ea[self.Na:]) ))
        
        FF_BB_mo  = np.einsum('ab, ij -> iajb', np.diag( self.Eb[self.Nb:]), np.eye( len( self.Eb[:self.Nb]) ))
        FF_BB_mo -= np.einsum('ij, ab -> iajb', np.diag( self.Eb[:self.Nb]), np.eye( len( self.Eb[self.Nb:]) ))
        
        
        A_AA = A_AAAA.swapaxes(0, 1) + FF_AA_mo
        A_AB = A_AABB.swapaxes(0, 1)
        A_BA = A_BBAA.swapaxes(0, 1)
        A_BB = A_BBBB.swapaxes(0, 1) + FF_BB_mo
        
        A_AA = A_AA.reshape( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) )
        A_AB = A_AB.reshape( ( len(self.Ea[self.Na:])*len(self.Ea[:self.Na]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) ) )
        A_BA = A_BA.reshape( ( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Ea[:self.Na]) ) )
        A_BB = A_BB.reshape( len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Eb[:self.Nb]) )
        
        self.A_UCIS = np.asarray( np.bmat([[A_AA, A_AB],[A_BA, A_BB]]) )
        
        return None
    
    def SFCIS(self):
        
        A_ABAB = -1.*np.einsum("Aa, Bi, Cj, Db, ADCB -> aijb", self.Cb[:, self.Nb:], self.Ca[:, :self.Na], self.Ca[:, :self.Na], self.Cb[:, self.Nb:], self.I, optimize=True)
        A_BABA = -1.*np.einsum("Aa, Bi, Cj, Db, ADCB -> aijb", self.Ca[:, self.Na:], self.Cb[:, :self.Nb], self.Cb[:, :self.Nb], self.Ca[:, self.Na:], self.I, optimize=True)
        
        ## Fock
        A_ABAB += np.einsum("ab, ij -> aijb", np.diag( self.Eb[self.Nb:] ), np.eye( len( self.Ea[:self.Na] ) ))
        A_ABAB -= np.einsum("ij, ab -> aijb", np.diag( self.Ea[:self.Na] ), np.eye( len( self.Eb[self.Nb:] ) ))
        
        A_BABA += np.einsum("ab, ij -> aijb", np.diag( self.Ea[self.Na:] ), np.eye( len( self.Eb[:self.Nb] ) ))
        A_BABA -= np.einsum("ij, ab -> aijb", np.diag( self.Eb[:self.Nb] ), np.eye( len( self.Ea[self.Na:] ) ))
        
        A_ABAB = A_ABAB.swapaxes(0, 1)
        A_BABA = A_BABA.swapaxes(0, 1)
        
        A_ABAB = A_ABAB.reshape( ( len(self.Ea[:self.Na])*len(self.Eb[self.Nb:]), len(self.Ea[:self.Na])*len(self.Eb[self.Nb:]) ) )
        A_BABA = A_BABA.reshape( ( len(self.Ea[self.Na:])*len(self.Eb[:self.Nb]), len(self.Ea[self.Na:])*len(self.Eb[:self.Nb]) ) )
        
        self.A_SF = np.asarray( np.bmat([[ A_ABAB, np.zeros(( A_ABAB.shape[1], A_BABA.shape[1] ))], [np.zeros(( A_BABA.shape[0], A_ABAB.shape[1] )),  A_BABA]]))
        
        B_BAAB  = -1.*np.einsum("Aa, Bi, Cb, Dj, ADCB -> aibj", self.Ca[:, self.Na:], self.Cb[:, :self.Nb], self.Cb[:, self.Nb:], self.Ca[:, :self.Na], self.I, optimize=True)
        B_ABBA  = -1.*np.einsum("Aa, Bi, Cb, Dj, ADCB -> aibj", self.Cb[:, self.Nb:], self.Ca[:, :self.Na], self.Ca[:, self.Na:], self.Cb[:, :self.Nb], self.I, optimize=True)
        
        B_BAAB = B_BAAB.reshape( ( len(self.Ea[self.Na:])*len(self.Eb[:self.Nb]), len(self.Eb[self.Nb:])*len(self.Ea[:self.Na]) ) , order="F")
        B_ABBA = B_ABBA.reshape( ( len(self.Eb[self.Nb:])*len(self.Ea[:self.Na]), len(self.Ea[self.Na:])*len(self.Eb[:self.Nb]) ) , order="F")
        
        self.B_SF = np.asarray( np.bmat([[ np.zeros(( B_ABBA.shape[0], B_BAAB.shape[1] )), B_ABBA], [B_BAAB,  np.zeros(( B_BAAB.shape[0], B_ABBA.shape[1] ))]]))
        self.SFHessian = 1.*( self.A_SF + self.B_SF )
        
        return None
    
    def CC_decompose(self):
        
        nbf = len(self.S)
        CA = np.einsum("Ai, Ba -> ABia", self.Ca[:, :self.Na], self.Ca[:, self.Na:]).reshape((nbf, nbf, self.Na * (nbf - self.Na )  ))
        CB = np.einsum("Ai, Ba -> ABia", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:]).reshape((nbf, nbf, self.Nb * (nbf - self.Nb )  ))
    
        return  np.concatenate((CA, CB), axis=2)
    
    def CCF_decompose(self):
        
        nbf = len(self.S)
        CA = np.einsum("Ai, Ba -> ABia", self.Ca[:, :self.Na], self.Ca[:, self.Na:]).reshape((nbf, nbf, self.Na * (nbf - self.Na )  ), order="F")
        CB = np.einsum("Ai, Ba -> ABia", self.Cb[:, :self.Nb], self.Cb[:, self.Nb:]).reshape((nbf, nbf, self.Nb * (nbf - self.Nb )  ), order="F")
    
        return  np.concatenate((CA, CB), axis=2)
    
    def UCIS_calc(self):
        
        w_I, X_IJ = np.linalg.eigh( self.A_UCIS ) #!!!
        TU_Ix = 1.0 * np.einsum("IJ, Jx -> Ix", X_IJ.T, np.einsum("xAB, ABI -> Ix", self.D, self.CC_decompose()))
        fU_I  = 2/3 * np.einsum("I, Ix, Ix -> I", w_I, TU_Ix, TU_Ix)
        
        self.CIS_E = w_I
        self.CIS_X = X_IJ.T
        self.CIS_f = fU_I
        
        return None
    
    def UCISF_calc(self):
        
        w_I, X_IJ = np.linalg.eigh( self.A_UCIS ) #!!!
        TU_Ix = 1.0 * np.einsum("IJ, Jx -> Ix", X_IJ.T, np.einsum("xAB, ABI -> Ix", self.D, self.CCF_decompose()))
        fU_I  = 2/3 * np.einsum("I, Ix, Ix -> I", w_I, TU_Ix, TU_Ix)
        
        self.CIS_E = w_I
        self.CIS_X = X_IJ.T
        self.CIS_f = fU_I
        
        return None
    
    def UTDHF_calc(self):
        
        R_UTDHF = (self.A_UCIS - self.B_TDHF) @ (self.A_UCIS + self.B_TDHF)
        L_UTDHF = (self.A_UCIS + self.B_TDHF) @ (self.A_UCIS - self.B_TDHF)
        
        Eigenvalues2, L_Eigenvectors = np.linalg.eig( L_UTDHF )
        Eigenvalues2, R_Eigenvectors = np.linalg.eig( R_UTDHF ) ## keep the order of R
        Eigenvalues = np.sqrt(Eigenvalues2)
        
        TU_Ix = 1.0*np.einsum("IJ, Jx -> Ix", R_Eigenvectors.T, np.einsum("xAB, ABI -> Ix", self.D, self.CC_decompose()))
        fU_I  = 2/3 * np.einsum("I, Ix, Ix -> I", Eigenvalues, TU_Ix, TU_Ix)
        
        self.TDHF_E = Eigenvalues
        self.TDHF_R = R_Eigenvectors.T
        self.TDHF_L = L_Eigenvectors.T
        self.TDHF_f = fU_I
        
        return None
    
    def RT_Ehrenfest(self):
        """
        Given Field + Atoms
        Sum Forces, f_ix, and Dipole-Spectrum d_tx
        """
        
        return None
    
    def getAO(self, DA, DB):
        ## transform MO density into an AO density 
        return ( (self.Ca)@ DA @((self.Ca).T) ), ( (self.Cb)@ DB @((self.Cb).T) )
    
    def getMO(self, FA, FB):
        ## transform AO Fock into an MO Fock 
        return ( ((self.Ca).T)@ FA @(self.Ca) ), ( ((self.Cb).T)@ FB @(self.Cb) )
    
    def RTHF_MMUTtqdm(self, DA_t=None, DB_t=None, dt = 0.002, dT = 1000, onoff=1.0, field=None, MD=False, Current=False, probe=False):

        tsteps = int(dT/dt)

        #if DA_t is None:
        DA_t = self.Da_mo
        #if DB_t is None:
        DB_t = self.Db_mo
        if field is None:
            def field(t):
                return np.zeros(3)

        ### compute initial half step propagators?
        DA_ao   = ( self.Ca ) @ (DA_t) @ ( (self.Ca).T)
        DB_ao   = ( self.Cb ) @ (DB_t) @ ( (self.Cb).T)
        FA_ao, FB_ao = self.get_UFock(DA_ao, DB_ao)
        FA0_mo  = ((self.Ca).T) @ FA_ao @ (self.Ca)
        FB0_mo  = ((self.Cb).T) @ FB_ao @ (self.Cb)
        UA_half = Expm( -1j*FA0_mo*dt/2 ) #!!!
        UB_half = Expm( -1j*FB0_mo*dt/2 ) #!!!
        DA_half = (UA_half.conj().T) @ DA_t @ (UA_half)
        DB_half = (UB_half.conj().T) @ DB_t @ (UB_half)

        if MD:
            f_ix = np.zeros(((self.dS).shape[0], (self.dS).shape[1]))
        if Current:
            J = np.zeros((tsteps, 2, len(self.Da_mo), len(self.Da_mo)), dtype=complex)
        if probe:
            energy = np.zeros(tsteps)
            trace  = np.zeros(tsteps)
        d_tx = np.zeros((tsteps, 3))
        for step in tqdm(range(tsteps)):
            t = step * dt

            # get AO density matrices
            DA_ao = ( self.Ca ) @ DA_t @ ( (self.Ca).T)
            DB_ao = ( self.Cb ) @ DB_t @ ( (self.Cb).T)

            # compute FA_t, FB_t
            FA_ao, FB_ao = self.get_UFock(DA_ao, DB_ao) - onoff * np.einsum('xmn, x -> mn', self.D, field.getEE(t) )

            # save to stuff
            d_tx[step] = np.einsum('xAB, AB -> x', self.D, ( DA_ao + DB_ao ).real )
            if Current:
                J[step, 0] = DA_t
                J[step, 1] = DB_t
            if probe:
                trace[step] = np.einsum("ii -> ", DA_t + DB_t).real
                energy[step] = self.get_Euhf(DA_ao, DB_ao, FA_ao, FB_ao).real
            if MD:
                f_ix += self.get_Cforce(DA_ao, DB_ao, FA_ao, FB_ao).real

            FA = ((self.Ca).T) @ FA_ao @ (self.Ca)
            FB = ((self.Cb).T) @ FB_ao @ (self.Cb)

            # compute propagators
            UA_half = Expm( -1j*FA*dt/2 ) #!!!
            UB_half = Expm( -1j*FB*dt/2 ) #!!!

            # half step forward
            DA_half = (UA_half) @ DA_half @ (UA_half.conj().T)
            DB_half = (UB_half) @ DB_half @ (UB_half.conj().T)

            # get AO density matrices
            DA_ao = ( self.Ca ) @ DA_half @ ( (self.Ca).T)
            DB_ao = ( self.Cb ) @ DB_half @ ( (self.Cb).T)

            # compute FA_t, FB_t
            t = (step + 0.5) * dt
            FA_ao, FB_ao = self.get_UFock(DA_ao, DB_ao) - onoff * np.einsum('xmn, x -> mn', self.D, field.getEE(t) )
            FA = ((self.Ca).T) @ FA_ao @ (self.Ca)
            FB = ((self.Cb).T) @ FB_ao @ (self.Cb)

            # compute propagators
            UA  = Expm( -1j*FA*dt ) #!!!
            UB  = Expm( -1j*FB*dt ) #!!!        

            # full step forward
            DA_t = (UA) @ DA_t @ (UA.conj().T)
            DB_t = (UB) @ DB_t @ (UB.conj().T)

        d_tx -= np.einsum("tx -> x", d_tx)/len(d_tx)
        if MD:
            f_ix  = f_ix/tsteps
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, f_ix, DA_t, DB_t
        if probe:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, trace, energy
        if Current:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, J
        else:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx
    
    def RTHF_UT(self, DA_t=None, DB_t=None, dt = 0.002, dT = 100, onoff=1.0, field=None, MD=False, Current=False, probe=False):
        tsteps = int(dT/dt)

        DA_t = 1.*(self.Da_mo)
        DB_t = 1.*(self.Db_mo)
        if field is None:
            field = E_field()
            field.E0 = 0.
            field.Γ  = 100.
            

        if MD:
            f_ix = np.zeros(((self.dS).shape[0], (self.dS).shape[1]))
        if Current:
            J = np.zeros((tsteps, 2, len(self.Da_mo), len(self.Da_mo)), dtype=complex)
        if probe:
            energy = np.zeros(tsteps)
            trace  = np.zeros(tsteps)
        d_tx   = np.zeros((tsteps, 3))
        for step in (range(tsteps)):
            t = (step) * dt

            # get AO density matrices
            DA_ao, DB_ao = self.getAO(DA_t, DB_t)
            d_tx[step]  = np.einsum('xAB, AB -> x', self.D, ( DA_ao + DB_ao ).real )

            # compute FA, FB and transform to MO basis
            FA_ao, FB_ao = self.get_UCFock(DA_ao, DB_ao) - onoff * np.einsum('xAB, x -> AB', self.D, field.getEE(t) )
            FA, FB = self.getMO(FA_ao, FB_ao)

            ####
            if Current:
                J[step, 0] = DA_t
                J[step, 1] = DB_t
            if probe:
                trace[step] = np.einsum("ii -> ", DA_t + DB_t).real
                energy[step] = self.get_Euhf(DA_ao, DB_ao, FA_ao, FB_ao).real
            if MD:
                f_ix += self.get_Cforce(DA_ao, DB_ao, FA_ao, FB_ao).real
            ####

            # compute propagators
            UA  = Expm( -1j*(dt)*FA )
            UB  = Expm( -1j*(dt)*FB )

            ### full step forward
            DA_t  = (UA) @ DA_t @ (UA.conj().T)
            DB_t  = (UB) @ DB_t @ (UB.conj().T)


        d_tx -= np.einsum("tx -> x", d_tx)/len(d_tx)

        if MD:
            f_ix  = f_ix/tsteps
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, f_ix, DA_t, DB_t
        if Current:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, J
        if probe:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, trace, energy
        else:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx
    
    def RTHF_RK4(self, DA_t=None, DB_t=None, dt = 0.002, dT = 100, onoff=1.0, field=None, MD=False, Current=False, probe=False):
        """ With Electric Dipole Interaction """
        """
        DA_t, DB_t = Initial MO Density (2d np.array)
        CA_T, CB_T = Coefficients in dT (2d np.array)
        dt     = Electronic Time Step (float)
        dT     = # of Electronic Time Steps (int)
        onoff  = Number to turn: off = 0 & on = 1 interaction (float)
        dipole = dipole integral in given direction (2d np.array)
        """
        tsteps = int(dT/dt)

        
        DA_t = 1.*(self.Da_mo)
        DB_t = 1.*(self.Db_mo)
        if field is None:
            field = E_field()
            field.E0 = 0.
            field.Γ  = 100.

        #time  = np.linspace(0, dT, int(dT/dt), endpoint=False)
        if MD:
            f_ix = np.zeros(self.dS.shape)
        if Current:
            J = np.zeros((tsteps, 2, len(self.Da_mo), len(self.Da_mo)), dtype=complex)
        if probe:
            energy = np.zeros(tsteps)
            trace  = np.zeros(tsteps)

        time = np.linspace(0, dt*dT, dT, endpoint=False)
        d_tx = np.zeros((tsteps, 3))
        for step in tqdm(range(tsteps)):
            # =============================================================================
            # Runge-Kutta 4th Order Integrator
            # =============================================================================
            # compute K1
            t = (step + 0.0) * dt
            # =============================================================================
            DA_ao, DB_ao = self.getAO(DA_t, DB_t)
            d_tx[step]  = np.einsum('xAB, AB -> x', self.D, ( DA_ao + DB_ao ).real )

            FA_ao, FB_ao = self.get_UCFock(DA_ao, DB_ao) - onoff * np.einsum('xAB, x -> AB', self.D, field.getEE(t) )

            if Current:
                J[step, 0] = DA_t
                J[step, 1] = DB_t
            if probe:
                trace[step] = np.einsum("ii -> ", DA_t + DB_t).real
                energy[step] = self.get_Euhf(DA_ao, DB_ao, FA_ao, FB_ao).real
            if MD:
                f_ix += self.get_force(DA_ao, DB_ao, FA_ao, FB_ao)

            F_t_αe = ((self.Ca).T) @ FA_ao @ (self.Ca) 
            F_t_βe = ((self.Cb).T) @ FB_ao @ (self.Cb)
            K1_αe = -1j*(F_t_αe@DA_t - DA_t@F_t_αe)
            K1_βe = -1j*(F_t_βe@DB_t - DB_t@F_t_βe)

            tempD_αe = DA_t + 0.5 * dt * K1_αe
            tempD_βe = DB_t + 0.5 * dt * K1_βe


            # compute K2
            t = (step + 0.5) * dt
            DA_ao, DB_ao = self.getAO(tempD_αe, tempD_βe)
            # =============================================================================
            FA_ao, FB_ao = self.get_UCFock(DA_ao, DB_ao) - onoff * np.einsum('xAB, x -> AB', self.D, field.getEE(t) )
            F_t_αe = ((self.Ca).T) @ FA_ao @ (self.Ca) 
            F_t_βe = ((self.Cb).T) @ FB_ao @ (self.Cb)
            K2_αe = -1j*(F_t_αe@tempD_αe - tempD_αe@F_t_αe)
            K2_βe = -1j*(F_t_βe@tempD_βe - tempD_βe@F_t_βe)

            tempD_αe = DA_t + 0.5 * dt * K2_αe
            tempD_βe = DB_t + 0.5 * dt * K2_βe


            # compute K3
            DA_ao, DB_ao = self.getAO(tempD_αe, tempD_βe)
            FA_ao, FB_ao = self.get_UCFock(DA_ao, DB_ao) - onoff * np.einsum('xAB, x -> AB', self.D, field.getEE(t) )
            F_t_αe = ((self.Ca).T) @ FA_ao @ (self.Ca) 
            F_t_βe = ((self.Cb).T) @ FB_ao @ (self.Cb)
            K3_αe = -1j*(F_t_αe@tempD_αe - tempD_αe@F_t_αe)
            K3_βe = -1j*(F_t_βe@tempD_βe - tempD_βe@F_t_βe)

            tempD_αe = DA_t + 1.0 * dt * K3_αe
            tempD_βe = DB_t + 1.0 * dt * K3_βe


            # compute K4
            t = (step + 1.0) * dt
            DA_ao, DB_ao = self.getAO(tempD_αe, tempD_βe)
            # =============================================================================
            FA_ao, FB_ao = self.get_UCFock(DA_ao, DB_ao) - onoff * np.einsum('xAB, x -> AB', self.D, field.getEE(t) )
            F_t_αe = ((self.Ca).T) @ FA_ao @ (self.Ca) 
            F_t_βe = ((self.Cb).T) @ FB_ao @ (self.Cb)
            K4_αe = -1j*(F_t_αe@tempD_αe - tempD_αe@F_t_αe)
            K4_βe = -1j*(F_t_βe@tempD_βe - tempD_βe@F_t_βe)

            DA_t += (dt/6.0) * (K1_αe + 2.0 * K2_αe + 2.0 * K3_αe + K4_αe) 
            DB_t += (dt/6.0) * (K1_βe + 2.0 * K2_βe + 2.0 * K3_βe + K4_βe)


            # =============================================================================
            # Output MO Currents
            # =============================================================================

        d_tx -= np.einsum("tx -> x", d_tx)/len(d_tx)
        
        if Current:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, J
        if MD:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, f_ix, DA_t, DB_t
        if probe:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx, trace, energy
        else:
            return np.linspace(0, dt*tsteps, tsteps, endpoint=False), d_tx
    
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
        return ( self.E0 * np.exp( - 4*np.log(2) * (t - self.t0)**2/(self.Γ**2) ) * np.sin( self.ω * (t - self.t0) + 1j * self.phase )) * (self.vector)
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