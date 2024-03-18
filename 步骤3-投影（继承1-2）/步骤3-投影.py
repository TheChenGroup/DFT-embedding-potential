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


mol_ref_coords = 'C2H6O.xyz'  # 初始体系的坐标文件（xyz文件）
Number_of_Atom = 9             # 初始体系的总原子数目
idx_A = [0, 8]          # “嵌入区域”原子在坐标文件中的index（从0开始计数）
DFT_type = 'RKS'                # DFT具体形势
xcfunc = 'hse06'                # 泛函
mol_basis = '6-31G'             # 基组
a_smear_sigma = 0           # “嵌入区域”的Gaussian smeaing width，如果之前smearing 顺利减小到0了基本，这里就可以将smearing设置为0了
mol_a_spin = 0                                                         # “嵌入区域”的自旋
mol_a_charge = -1                                                      # “嵌入区域”的电荷
pc_coords_a = np.array(np.loadtxt('charge_A.txt').reshape(-1,3))       # “嵌入区域”的capping charge的xyz坐标文件“charge_A.txt”,在有多个电荷的时候换行排列就行，最后都是N行3列
pc_charges_a = np.array([1 for i in range(len(pc_coords_a))])          # “嵌入区域”的capping charge的每个charge带电多少，这里默认为+1
miu = 10000                     # 投影的energy-shift参数

# 可以先不修改的默认项
shutil.copyfile(mol_ref_coords, "part_A.xyz")                         # 根据上面设置的idx_A，自动生成的“嵌入区域”的坐标文件“part_A.xyz”，其中“环境”区域原子都用Ghost atoms替代了
conv_tol = 1e-11                # DFT计算过程中的能量收敛精度
conv_tol_grad = 1e-7            # DFT计算过程中的gradient收敛精度


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
        vmat_ext = vmat_ext.reshape(len(ha_ref) , len(ha_ref))
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
    occ = mf.get_occ()
    if coeffname is not None:
        numpy.savetxt(coeffname, mf.mo_coeff[:,mf.mo_occ>0.5]) # 只取占据轨道的mo_coeff    
    if moldenname is not None:
        molden.from_scf(mf, moldenname)
    return E, dm, h, s, occ
    

mol_a_coords = choose_atoms(idx_A, 'part_A.xyz')

Vemb = numpy.loadtxt('V_DMFET.txt')
dma = numpy.loadtxt('dma.txt')
dmb = numpy.loadtxt('dmb.txt')
ha_ref = numpy.loadtxt('ha_ref.txt')
s_ref = numpy.loadtxt('s_ref.txt')

# 构造投影算符P，将miu*P加在Vemb上做计算，得到最终的结果
Projector = s_ref.dot(dmb).dot(s_ref)
Vemb_with_P = (Vemb + miu * Projector).reshape(len(ha_ref) , len(ha_ref))
Ea_P, dma_P, ha_P, sa_P, occa_P = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname = 'a-vext_P-coeff.txt', smear_sigma = a_smear_sigma, dm_init_guess = np.loadtxt('dma.txt'), pc_coords = pc_coords_a, pc_charges = pc_charges_a, vmat_ext = Vemb_with_P, h1 = ha_ref, moldenname = 'a-vext_P.molden')
numpy.savetxt('Vemb_with_P.txt', Vemb_with_P)        # 这个就是最终加了投影的势能
numpy.savetxt('dma_P.txt', dma_P)
numpy.savetxt('occa_P.txt', occa_P)

print('dma_deta Max = ', max(abs((dma_P - dma).reshape(1,-1)[0])))    # 看一下加了投影后的嵌入区域的密度矩阵，与没有加投影的密度矩阵的变化，应该非常小才对
