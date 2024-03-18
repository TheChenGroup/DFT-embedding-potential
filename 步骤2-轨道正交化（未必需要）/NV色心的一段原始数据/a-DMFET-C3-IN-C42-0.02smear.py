# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:02:32 2021

@author: 19011
"""

import numpy as np
import numpy
import pyscf
from functools import reduce
from pyscf import gto, scf, dft, qmmm
from pyscf import lo
from pyscf.tools import molden
from pyscf.scf import atom_hf
from pyscf.dft import numint
import matplotlib.pyplot as plt
import time
from scipy import optimize
import os
import shutil

mol_reff_coords = 'C42H42N.xyz'
Number_of_Atom = 85
idx_A = [0, 9, 12, 27]
DFT_type = 'RKS'
xcfunc = 'hse06'
mol_basis = '6-31G'
a_smear_sigma = 0.02
OvlpCutValue = 0.01
conv_tol = 1e-11
conv_tol_grad = 1e-7

mol_reff_spin = 0
mol_reff_charge = -1
mol_a_spin = 0
mol_a_charge = -13 
pc_coords_a = np.array(np.loadtxt('charge_A.txt').reshape(-1,3))
pc_charges_a = np.array([1 for i in range(len(pc_coords_a))])

shutil.copyfile(mol_reff_coords, "part_A.xyz")


number_iter = 0

def choose_atoms(atom_index, xyzfile):
# Choose Ghost Atoms, atom_index is the index of nonghost-atom
    with open(xyzfile, 'r') as f:
        xyz = f.readlines()
        xyz_title = xyz[:2]
        xyz_coord = xyz[2:]
        atom_tot = [j for j in range(len(xyz_coord))]
        for k in atom_tot:
            if k not in atom_index:
                xyz_coord[k] = 'ghost-' + xyz_coord[k]
    new_xyz = xyz_title + xyz_coord
    
    with open(xyzfile, 'w') as g:
        for j in new_xyz:
            g.write(j)
    return xyzfile


def DFT_cal(mol_coords, mol_spin, mol_charge, smear_sigma, coeffname = None, dm_init_guess = None, moldenname = None, pc_coords = None, pc_charges = None, vmat_ext = None, h1 = None, mol_ecp = None):
# DFT Calculator   
    global DFT_type, conv_tol, conv_tol_grad, xcfunc, DFT_type, mol_basis
    ni = dft.numint.NumInt()
    mol = gto.Mole()
    mol.atom = mol_coords
    mol.spin = mol_spin
    mol.charge = mol_charge
    mol.basis = mol_basis
#    mol.unit = mol_unit
    if mol_ecp is not None:
        mol.ecp = mol_ecp    
    mol.build()  
    
    if DFT_type == 'RKS':
        mf = dft.RKS(mol)
    elif DFT_type == 'ROKS':
        mf = dft.ROKS(mol)
    elif DFT_type == 'UKS':
        mf = dft.UKS(mol)
    else:
        print('Please input DFTtype as RKS/ROKS/UKS')
    
    if (pc_coords is not None) and (pc_charges is not None):        
        mf = qmmm.mm_charge(mf, pc_coords, pc_charges)

    if (h1 is not None) and (vmat_ext is not None):    
        vmat_ext = vmat_ext.reshape(len(ha_reff) , len(ha_reff))
        h2 = h1 + vmat_ext        
        mf.get_hcore = lambda *args:h2        

    mf.occ_path = os.getcwd()      
    mf.max_cycle = 200
#    mf.verbose = 5
    mf.xc = xcfunc
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.smear_sigma = smear_sigma
#    mf = mf.newton()

    if dm_init_guess is not None:
        E = mf.kernel(dm0 = dm_init_guess)
    else:
        E = mf.kernel()  
        
    dm = mf.make_rdm1() 
    h = mf.get_hcore()   
    s = mf.get_ovlp(mol)
    if coeffname is not None:
        numpy.savetxt(coeffname, mf.mo_coeff[:,mf.mo_occ>0.5]) # 只取占据轨道的mo_coeff    
    if moldenname is not None:
        molden.from_scf(mf, moldenname)
    return E, dm, h, s
    

def Lagrangian(vmat_ao):
# Get Functional and Gradient
    global OvlpCutValue, s_reff, number_iter, ha_reff, mol_a_coords, mol_a_spin, mol_a_charge, pc_coords_a, pc_charges_a   
    vmat_ao = vmat_ao.reshape(len(ha_reff) , len(ha_reff))
    if number_iter > -1:
        E_a, dm_a, h_a, s_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname = 'a-vext-coeff.txt', smear_sigma = a_smear_sigma, dm_init_guess = np.loadtxt('dma.txt'), pc_coords = pc_coords_a, pc_charges = pc_charges_a, vmat_ext = vmat_ao, h1 = ha_reff, moldenname = 'a-vext.molden')
    else:    
        E_a, dm_a, h_a, s_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname = 'a-vext-coeff.txt', smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, vmat_ext = vmat_ao, h1 = ha_reff, moldenname = 'a-vext.molden')

    grad_dma = -(dm_a - Pa_new)
    Wa = -(E_a - np.sum(vmat_ao * Pa_new))
    grad_dma = np.array(grad_dma.reshape(1,-1)[0])
    print('grad_max_a = ', max(abs(grad_dma)))
    print('Wa = ', Wa)
    numpy.savetxt('vmat_ao.txt', vmat_ao)
    numpy.savetxt('dma.txt', dm_a)
    numpy.savetxt('sa.txt', s_a)

    a_vext_coeff = np.loadtxt('a-vext-coeff.txt') 
    New_coeff_B = np.loadtxt('New_coeff_B.txt')
    MO_ovlp_vext = np.dot(a_vext_coeff.T, s_reff).dot(New_coeff_B)
    print('New Max Abs Ovlp Value = ', np.max(abs(MO_ovlp_vext)))
    print('shape of a-vext-coeff = ', np.shape(a_vext_coeff))
    print('shape of New_coeff_B = ', np.shape(New_coeff_B))
    np.savetxt('MO_ovlp_vext.txt', MO_ovlp_vext)
    print('shape of MO-vext-ovlp = ', np.shape(MO_ovlp_vext))
    for i in range(len(MO_ovlp_vext)):
        for j in range(len(MO_ovlp_vext[i])):
            if abs(MO_ovlp_vext[i][j]) > OvlpCutValue:
                print('Ovlp of MO ', i+1, ' & ', j+1, ' = ', MO_ovlp_vext[i][j])
    
    number_iter += 1
    print('N_iteration = ', number_iter)
    return Wa, grad_dma


mol_a_coords = choose_atoms(idx_A, 'part_A.xyz')

Pa_new = numpy.loadtxt('Pa_new.txt')
ha_reff = numpy.loadtxt('ha_reff.txt')
sa_reff = numpy.loadtxt('sa_reff.txt')
s_reff = numpy.loadtxt('s_reff.txt')


#vmat_ao0 = np.zeros(len(h_reff) * len(h_reff))
vmat_ao0 = np.loadtxt('vmat_ao-0.02.txt').reshape(1,-1)
x_final,w_final,d = optimize.fmin_l_bfgs_b( Lagrangian, x0=vmat_ao0, args=(), factr=1e4, pgtol=1e-05, maxls=1000) 
#x_final,w_final,d = optimize.minimize( Lagrangian, x0=vmat_ao0, args=(), method='CG')
print(x_final)
print(w_final)
print(d)
