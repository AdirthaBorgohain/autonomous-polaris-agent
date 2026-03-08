"""Molecular featurization: Morgan fingerprints and RDKit descriptors."""

from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray


def _smiles_to_mol(smiles: str):
    if isinstance(smiles, bytes):
        smiles = smiles.decode("utf-8")
    return Chem.MolFromSmiles(str(smiles).strip())


def morgan_fingerprints(
    smiles_list: Union[List[str], np.ndarray],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan (ECFP) bit vector for each SMILES. Invalid mols -> zero vector."""
    arr = np.asarray(smiles_list).ravel()
    n = len(arr)
    out = np.zeros((n, n_bits), dtype=np.float32)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i in range(n):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        fp = mfpgen.GetFingerprint(mol)
        ConvertToNumpyArray(fp, out[i])
    return out


def rdkit_descriptors(smiles_list: Union[List[str], np.ndarray]) -> np.ndarray:
    """Compute a small set of RDKit descriptors per SMILES. Invalid mols -> NaN row."""
    desc_fns = [
        Descriptors.MolLogP,
        Descriptors.MolWt,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
]
    arr = np.asarray(smiles_list).ravel()
    n = len(arr)
    out = np.full((n, len(desc_fns)), np.nan, dtype=np.float32)
    for i in range(n):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        for j, fn in enumerate(desc_fns):
            try:
                out[i, j] = fn(mol)
            except Exception:
                pass
    # Impute NaN with column median for sklearn compatibility
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = ~np.isnan(col)
        if mask.any():
            out[~mask, j] = np.median(col[mask])
        else:
            out[:, j] = 0.0
    return out


def build_features(
    inputs,
    fingerprint: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Build feature matrix from benchmark inputs (SMILES) and fingerprint type."""
    if hasattr(inputs, "__iter__") and not isinstance(inputs, str):
        smiles_list = list(inputs)
    else:
        smiles_list = [inputs]
    fingerprint = (fingerprint or "morgan").lower().strip()
    if fingerprint in ("morgan", "ecfp", "morgan_fingerprint"):
        return morgan_fingerprints(smiles_list, radius=radius, n_bits=n_bits)
    if fingerprint in ("rdkit_descriptors", "descriptors", "rdkit"):
        return rdkit_descriptors(smiles_list)
    return morgan_fingerprints(smiles_list, radius=radius, n_bits=n_bits)
