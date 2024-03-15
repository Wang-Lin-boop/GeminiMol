import numpy as np
from rdkit import Chem
import rdkit.Chem.rdFMCS as FMCS
from rdkit.Chem import SaltRemover
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

def cal_MCS_score(smiles1, smiles2, atom_mode="any"):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    mols = [mol1, mol2]
    min_bond_num = np.min([len(mol1.GetBonds()), len(mol2.GetBonds())])
    res = FMCS.FindMCS(mols, ringMatchesRingOnly=True, atomCompare=(FMCS.AtomCompare.CompareAny if atom_mode == "any" else FMCS.AtomCompare.CompareElements))
    return 0.0 if res is None else res.numBonds / min_bond_num

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except ValueError:
        return False
    except TypeError:
        raise TypeError(f"{smiles} caused TypeError.")
    if mol is None:
            return False
    return True

def check_smiles_validity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        return smiles
    except:
        return 'smiles_unvaild'

def is_single_ring_system(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Check if the molecule has more than 3 atoms
    if mol.GetNumAtoms() < 4:
        return True
    # Get the symmetrical SSSR and the atoms in the rings
    sssr = Chem.GetSymmSSSR(mol)
    ring_atoms = set()
    for ring in sssr:
        ring_atoms |= set(ring)
    # Check if the number of atoms in the rings is equal to the total number of atoms
    return len(ring_atoms) == mol.GetNumAtoms()

def gen_smiles_list(smiles, expand_ratio):
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    smiles_list = [smiles]
    try:
        smiles_list += Chem.rdmolfiles.MolToRandomSmilesVect(mol, kekuleSmiles=False,numSmiles=expand_ratio, isomericSmiles=True)
    except:
        for _ in range(expand_ratio):
            try:
                smiles_list.append(Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=True, isomericSmiles=True))
            except:
                print(smiles)
    return list(set(smiles_list))

def gen_smiles(smiles, kekule=False, random=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random, isomericSmiles=True)
    except:
        smiles = 'smiles_unvaild'
    return smiles

def gen_standardize_smiles(smiles, kekule=False, random=False):
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        if mol is None:
            return 'smiles_unvaild'
        desalt = SaltRemover.SaltRemover() ## defnData="[Cl,Br,I,Fe,Na,K,Ca,Mg,Ni,Zn]"
        mol = desalt.StripMol(mol)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random, isomericSmiles=True)
        return smiles
    except:
        smiles = 'smiles_unvaild'

def smiles2len(smiles):
    return len(smiles)

def smiles2mw(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = Descriptors.MolWt(mol)
    except:
        mol_weight = 'smiles_unvaild'
    return mol_weight

def smiles2HA(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        HA_num = mol.GetNumHeavyAtoms()
    except:
        HA_num = 'smiles_unvaild'
    return HA_num

def smiles2RingNum(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Ring_num = mol.GetRingInfo().NumRings()
    except:
        Ring_num = 'smiles_unvaild'
    return Ring_num

def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def smiles2RS(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        RS_num = len(GetRingSystems(mol))
    except:
        RS_num = 'smiles_unvaild'
    return RS_num

def rule_of_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    nrb = Descriptors.NumRotatableBonds(mol)
    # psa = Descriptors.TPSA(mol)
    if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and logp >= -2 and nrb <= 10):
        return 1
    else:
        return 0

def get_skeleton(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        skeleton = Chem.MolToSmiles(scaffold)
        return skeleton
    except:
        return 'smiles_unvaild'



