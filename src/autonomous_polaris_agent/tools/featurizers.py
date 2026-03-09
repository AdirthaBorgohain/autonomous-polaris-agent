"""Molecular featurization: Morgan fingerprints and RDKit descriptors."""

from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray


def _smiles_to_mol(smiles: str):
    if isinstance(smiles, bytes):
        smiles = smiles.decode("utf-8")
    return Chem.MolFromSmiles(str(smiles).strip())


def _prep_fingerprint_out(
    smiles_list: Union[List[str], np.ndarray], n_bits: int
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(smiles_list).ravel()
    out = np.zeros((len(arr), n_bits), dtype=np.float32)
    return arr, out


def morgan_fingerprints(
    smiles_list: Union[List[str], np.ndarray],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan (ECFP) bit vector for each SMILES. Invalid mols -> zero vector."""
    arr, out = _prep_fingerprint_out(smiles_list=smiles_list, n_bits=n_bits)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i in range(len(arr)):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        fp = mfpgen.GetFingerprint(mol)
        ConvertToNumpyArray(fp, out[i])
    return out


def morgan_count_fingerprints(
    smiles_list: Union[List[str], np.ndarray],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan count vector for each SMILES. Invalid mols -> zero vector."""
    arr, out = _prep_fingerprint_out(smiles_list=smiles_list, n_bits=n_bits)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i in range(len(arr)):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        fp = mfpgen.GetCountFingerprint(mol)
        nonzero = fp.GetNonzeroElements()
        for idx, count in nonzero.items():
            if 0 <= idx < n_bits:
                out[i, idx] = float(count)
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


def maccs_fingerprints(smiles_list: Union[List[str], np.ndarray]) -> np.ndarray:
    """Compute MACCS 167-bit keys per SMILES. Invalid mols -> zero vector."""
    n_bits = 167
    arr, out = _prep_fingerprint_out(smiles_list=smiles_list, n_bits=n_bits)
    for i in range(len(arr)):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        ConvertToNumpyArray(fp, out[i])
    return out


def atom_pair_fingerprints(
    smiles_list: Union[List[str], np.ndarray],
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute atom pair fingerprint per SMILES. Invalid mols -> zero vector."""
    arr, out = _prep_fingerprint_out(smiles_list=smiles_list, n_bits=n_bits)
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    for i in range(len(arr)):
        mol = _smiles_to_mol(arr[i])
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        ConvertToNumpyArray(fp, out[i])
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
    if fingerprint in ("morgan_counts",):
        return morgan_count_fingerprints(
            smiles_list, radius=radius, n_bits=n_bits
        )
    if fingerprint in ("morgan_rdkit_concat",):
        morgan_part = morgan_fingerprints(
            smiles_list, radius=radius, n_bits=n_bits
        )
        rdkit_part = rdkit_descriptors(smiles_list)
        return np.hstack([morgan_part, rdkit_part])
    if fingerprint in ("rdkit_descriptors", "descriptors", "rdkit"):
        return rdkit_descriptors(smiles_list)
    if fingerprint in ("maccs",):
        return maccs_fingerprints(smiles_list)
    if fingerprint in ("atom_pair", "atompair"):
        return atom_pair_fingerprints(smiles_list, n_bits=n_bits)
    return morgan_fingerprints(smiles_list, radius=radius, n_bits=n_bits)
