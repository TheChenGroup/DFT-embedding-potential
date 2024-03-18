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
#import matplotlib.pyplot as plt
import time
from scipy import optimize
import os
import shutil

isFristOEP = True             # 是否是第一次做这个计算的DMFET（有无Vemb继承），True代表是第一次，以后续算都是False
mol_ref_coords = 'C2H6O.xyz'  # 初始体系的坐标文件（xyz文件）
Number_of_Atom = 9             # 初始体系的总原子数目
idx_A = [0, 8]          # “嵌入区域”原子在坐标文件中的index（从0开始计数）
DFT_type = 'RKS'                # DFT具体形势
xcfunc = 'hse06'                # 泛函
mol_basis = '6-31G'             # 基组
total_smear_sigma = 0.02        # 总系统的Gaussian smeaing width，没有smearing时设为0
a_smear_sigma = 0.02            # “嵌入区域”的Gaussian smeaing width
b_smear_sigma = 0.02            # “环境”的Gaussian smeaing width
mol_ref_spin = 0                                                       # 总系统的自旋
mol_ref_charge = 0                                                     # 总系统的电荷
mol_a_spin = 0                                                         # “嵌入区域”的自旋
mol_a_charge = -1                                                      # “嵌入区域”的电荷
pc_coords_a = np.array(np.loadtxt('charge_A.txt').reshape(-1,3))       # “嵌入区域”的capping charge的xyz坐标文件“charge_A.txt”,在有多个电荷的时候换行排列就行，最后都是N行3列
pc_charges_a = np.array([1 for i in range(len(pc_coords_a))])          # “嵌入区域”的capping charge的每个charge带电多少，这里默认为+1
mol_b_spin = 0                                                         # “环境”的自旋
mol_b_charge = +1                                                      # “环境”的电荷（注意与“嵌入区域”电荷之和，应该为总系统的电荷）
pc_coords_b = np.array(np.loadtxt('charge_B.txt').reshape(-1,3))       # “环境”的capping charge的坐标文件“charge_B.txt”
pc_charges_b = np.array([-1 for i in range(len(pc_coords_b))])         # “环境”的capping charge的每个charge带电多少，这里默认为-1

# 可以先不修改的默认项
shutil.copyfile(mol_ref_coords, "part_A.xyz")                         # 根据上面设置的idx_A，自动生成的“嵌入区域”的坐标文件“part_A.xyz”，其中“环境”区域原子都用Ghost atoms替代了
shutil.copyfile(mol_ref_coords, "part_B.xyz")                         # 根据上面设置的idx_A，自动生成的“环境”的坐标文件“part_B.xyz”，其中“嵌入”区域原子都用Ghost atoms替代了
number_iter = 0                 # 当前进行的DMFET的OEP的迭代次数
OvlpCutValue = 0.01             # 人为设置的“交叠轨道空间”的阈值，默认为0.01
conv_tol = 1e-11                # DFT计算过程中的能量收敛精度
conv_tol_grad = 1e-7            # DFT计算过程中的gradient收敛精度


def choose_atoms(atom_index, xyzfile):                                 # 构造“part_A.xyz”和“part_B.xyz”的函数
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
# 这里有大量的可选参数
# coeffname代表是否需要输出分子轨道的coefficient，需要的话命名
# dm_init_guess是当前DMFET开始时的初始猜测，在第一次做DMFET时为默认值None，在以后续算时，需要继承上次得到的density_matrix，在这里输入
# moldenname代表是否需要输出轨道molden文件，需要的话命名
# pc_coords是capping charges的坐标文件
# pc_charges是capping charges的电荷，是一个和pc_coords里序号一一对应的数组
# vmat_ext就是Vemb外场的matrix
# h1是哈密顿量的单电子项h1
# mol_ecp是是否使用赝势，是的话请指定类型（参见PySCF）
# 函数输出为 E：能量；dm：密度矩阵；h：单电子项h1；s：原子轨道的Ovlp矩阵，occ：分子轨道占据数

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
        vmat_ext = vmat_ext.reshape(len(h_ref) , len(h_ref))
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
    

def Lagrangian(V_DMFET): # OEP过程
# Get Functional and Gradient
    global isFristOEP, OvlpCutValue, s_ref, number_iter, ha_ref, hb_ref, mol_a_coords, mol_b_coords, mol_a_spin, mol_a_charge, mol_b_spin, mol_b_charge, pc_coords_a, pc_charges_a, pc_coords_b, pc_charges_b    
    V_DMFET = V_DMFET.reshape(len(h_ref) , len(h_ref))
    if (isFristOEP and number_iter == 0):
        E_a, dm_a, h_a, s_a, occ_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname = 'a-vext-coeff.txt', smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, vmat_ext = V_DMFET, h1 = ha_ref, moldenname = 'a-vext.molden')
        E_b, dm_b, h_b, s_b, occ_b = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, coeffname = 'b-vext-coeff.txt', smear_sigma = b_smear_sigma, pc_coords = pc_coords_b, pc_charges = pc_charges_b, vmat_ext = V_DMFET, h1 = hb_ref, moldenname = 'b-vext.molden') 
    else:    
        E_a, dm_a, h_a, s_a, occ_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname = 'a-vext-coeff.txt', smear_sigma = a_smear_sigma, dm_init_guess = np.loadtxt('dma.txt'), pc_coords = pc_coords_a, pc_charges = pc_charges_a, vmat_ext = V_DMFET, h1 = ha_ref, moldenname = 'a-vext.molden')
        E_b, dm_b, h_b, s_b, occ_b = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, coeffname = 'b-vext-coeff.txt', smear_sigma = b_smear_sigma, dm_init_guess = np.loadtxt('dmb.txt'), pc_coords = pc_coords_b, pc_charges = pc_charges_b, vmat_ext = V_DMFET, h1 = hb_ref, moldenname = 'b-vext.molden')         

    grad_dm = -(dm_a + dm_b - dm_ref)
    W = -(E_a + E_b - np.sum(V_DMFET * dm_ref))
    grad_dm = np.array(grad_dm.reshape(1,-1)[0])
    print('grad_max = ', max(abs(grad_dm)))         # 当前gradient是多少
    print('W = ', W)                                # 当前泛函W是多少
    numpy.savetxt('V_DMFET.txt', V_DMFET)           # 存储当前的嵌入势"V_DMFET.txt"
    numpy.savetxt('dma.txt', dm_a)                  # 存储当前的"嵌入区域"的密度矩阵"dma.txt"
    numpy.savetxt('dmb.txt', dm_b)                  # 存储当前的"环境"的密度矩阵"dmb.txt"
    numpy.savetxt('occa.txt', occ_a)                # 存储当前的"嵌入区域"的分子轨道占据数"occa.txt"
    numpy.savetxt('occb.txt', occ_b)                # 存储当前的"嵌入区域"的分子轨道占据数"occb.txt"

    a_vext_coeff = np.loadtxt('a-vext-coeff.txt')                             # 存储当前的"嵌入区域"的分子轨道系数"a-vext-coeff.txt"
    b_vext_coeff = np.loadtxt('b-vext-coeff.txt')                             # 存储当前的"环境"的分子轨道系数"b-vext-coeff.txt"
    MO_ovlp_vext = np.dot(a_vext_coeff.T, s_ref).dot(b_vext_coeff)
    print('Max Abs Ovlp Value = ', np.max(abs(MO_ovlp_vext)))                 # 当前“嵌入区域”和"环境"的分子轨道的最大交叠值为多少？
    print('shape of a-vext-coeff = ', np.shape(a_vext_coeff))
    print('shape of b-vext-coeff = ', np.shape(b_vext_coeff))
    np.savetxt('MO_ovlp_vext.txt', MO_ovlp_vext)                              # 存储当前的“嵌入区域”和"环境"的分子轨道的最大交叠值矩阵为'MO_ovlp_vext.txt'
    print('shape of MO-vext-ovlp = ', np.shape(MO_ovlp_vext))
    for i in range(len(MO_ovlp_vext)):
        for j in range(len(MO_ovlp_vext[i])):
            if abs(MO_ovlp_vext[i][j]) > OvlpCutValue:
                print('Ovlp of MO ', i+1, ' & ', j+1, ' = ', MO_ovlp_vext[i][j])    # 大于设定阈值的分子轨道的序号分别是？第一个序号为"嵌入区域"，第二个序号为"环境"
    
    number_iter += 1
    print('N_iteration = ', number_iter)                                      # 当前OEP的迭代次数
    return W, grad_dm

atom_idx = [i for i in range(Number_of_Atom)]
idx_B = []
for j in atom_idx:
    if j not in idx_A:
        idx_B.append(j)
mol_a_coords = choose_atoms(idx_A, 'part_A.xyz')
mol_b_coords = choose_atoms(idx_B, 'part_B.xyz')




# 从这里开始设定OEP，如果是第一次做DMFET，dm_init_guess不用设置，如果是续算OEP，则以下都需要设置
# 先是总的系统做一次DFT，再是“嵌入区域”做一次DFT(为了得到没有嵌入势的ha)，最后是“环境”做一次DFT(为了得到没有嵌入势的hb)
# OEP第一次开始的话，Vemb的初始猜测为全0矩阵
# 不是第一次开始的话，继承上一次得到的Vemb，文件名可以自己定，这里是'V_DMFET.txt'

if isFristOEP:
    E_ref, dm_ref, h_ref, s_ref, occ_ref = DFT_cal(mol_ref_coords, mol_ref_spin, mol_ref_charge, smear_sigma = total_smear_sigma, moldenname = 'ref.molden', dm_init_guess = None) 
    Ea_ref, dma_ref, ha_ref, sa_ref, occa_ref = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, moldenname = 'a-ref.molden', dm_init_guess = None)
    Eb_ref, dmb_ref, hb_ref, sb_ref, occb_ref = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, smear_sigma = b_smear_sigma, pc_coords = pc_coords_b, pc_charges = pc_charges_b, moldenname = 'b-ref.molden', dm_init_guess = None)    
    V_DMFET0 = np.zeros(len(h_ref) * len(h_ref))
else:
    E_ref, dm_ref, h_ref, s_ref, occ_ref = DFT_cal(mol_ref_coords, mol_ref_spin, mol_ref_charge, smear_sigma = total_smear_sigma, moldenname = 'ref.molden', dm_init_guess = np.loadtxt('dm_ref.txt')) 
    Ea_ref, dma_ref, ha_ref, sa_ref, occa_ref = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, moldenname = 'a-ref.molden', dm_init_guess = np.loadtxt('dma_ref.txt'))
    Eb_ref, dmb_ref, hb_ref, sb_ref, occb_ref = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, smear_sigma = b_smear_sigma, pc_coords = pc_coords_b, pc_charges = pc_charges_b, moldenname = 'b-ref.molden', dm_init_guess = np.loadtxt('dmb_ref.txt'))
    V_DMFET0 = np.loadtxt('V_DMFET.txt').reshape(1,-1)

# 所有带"ref"的都是“参考”的，即“没有加嵌入势的”
numpy.savetxt('dm_ref.txt', dm_ref)
numpy.savetxt('h_ref.txt', h_ref)
numpy.savetxt('s_ref.txt', s_ref)
numpy.savetxt('ha_ref.txt', ha_ref)
numpy.savetxt('hb_ref.txt', hb_ref)
numpy.savetxt('dma_ref.txt', dma_ref)
numpy.savetxt('dmb_ref.txt', dmb_ref)

# Scipy的L-BFG-S算法迭代
x_final,w_final,d = optimize.fmin_l_bfgs_b( Lagrangian, x0=V_DMFET0, args=(), factr=1e4, pgtol=1e-05, maxls=1000) 
#x_final,w_final,d = optimize.minimize( Lagrangian, x0=V_DMFET0, args=(), method='CG')
print(x_final)
print(w_final)
print(d)