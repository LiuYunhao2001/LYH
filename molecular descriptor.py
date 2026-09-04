import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, Fragments, MACCSkeys
from rdkit import RDLogger
import warnings

warnings.filterwarnings('ignore')

INPUT_EXCEL = "Y.xlsx"
OUTPUT_CSV = "molecular_features.csv"
ECFP_RADIUS = 2
ECFP_BITS = 512
MACCS_BITS = 167
TARGET_SMILES_COL = "smiles"


def check_smiles_column(df, target_col):
    df_cols = [col.strip().lower() for col in df.columns]
    target_col_lower = target_col.strip().lower()
    match_cols = [
        df.columns[i] for i, col in enumerate(df_cols)
        if target_col_lower in col or col in target_col_lower
    ]
    if not match_cols:
        raise ValueError(
            f"SMILES column not found.\nAvailable columns: {df.columns.tolist()}\nExpected: {target_col}"
        )
    elif len(match_cols) > 1:
        print(f"Multiple SMILES columns found, using first: {match_cols[0]}")
        return match_cols[0]
    else:
        print(f"SMILES column matched: {match_cols[0]}")
        return match_cols[0]


def calculate_2d_descriptors(mol):
    selected_desc_funcs = [
        Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
        Lipinski.NumHDonors, Lipinski.NumHAcceptors, Descriptors.NumRotatableBonds,
        Descriptors.RingCount, Descriptors.HeavyAtomCount, Descriptors.FractionCSP3,
        Descriptors.Chi0v, Descriptors.Chi1v, Descriptors.Chi2v,
        Descriptors.BCUT2D_MWHI, Descriptors.BCUT2D_CHGHI, Descriptors.LabuteASA,
        Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
        Descriptors.NumHeteroatoms, Descriptors.NumAmideBonds,
        Descriptors.NumBridgeheadAtoms, Descriptors.NumAtomStereoCenters,
        Descriptors.NumUnspecifiedAtomStereoCenters, Descriptors.qed,
        Descriptors.MaxEStateIndex, Descriptors.MinEStateIndex, Descriptors.HeavyAtomMolWt,
        Descriptors.NHOHCount, Descriptors.NOCount, Fragments.fr_Al_OH,
        Fragments.fr_Ar_OH, Fragments.fr_COO, Fragments.fr_NH0,
        Fragments.fr_NH1, Fragments.fr_NH2, Fragments.fr_N_O,
        Fragments.fr_nitro, Fragments.fr_halogen
    ]
    return [func(mol) for func in selected_desc_funcs]


def calculate_fingerprint(mol, fp_type="ecfp4"):
    if fp_type == "ecfp4":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_BITS)
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError("Only ecfp4 or maccs supported")
    return list(fp)


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')
    print("Molecular feature calculation started")

    df = pd.read_excel(INPUT_EXCEL)
    print(f"Excel loaded. Total molecules: {len(df)}")

    smiles_col = check_smiles_column(df, TARGET_SMILES_COL)
    smiles_list = df[smiles_col].astype(str).values

    desc_names = [
        'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', 'FractionCSP3',
        'Chi0v', 'Chi1v', 'Chi2v', 'BCUT2D_MWHI', 'BCUT2D_CHGHI', 'LabuteASA',
        'NumAromaticRings', 'NumAliphaticRings', 'NumHeteroatoms', 'NumAmideBonds',
        'NumBridgeheadAtoms', 'NumAtomStereoCenters', 'NumUnspecifiedAtomStereoCenters',
        'QED', 'MaxEStateIndex', 'MinEStateIndex', 'HeavyAtomMolWt', 'NHOHCount',
        'NOCount', 'fr_Al_OH', 'fr_Ar_OH', 'fr_COO', 'fr_NH0', 'fr_NH1', 'fr_NH2',
        'fr_N_O', 'fr_nitro', 'fr_halogen'
    ]
    desc_records = []
    desc_valid = []

    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
            desc_records.append(calculate_2d_descriptors(mol))
            desc_valid.append(True)
        except Exception:
            desc_records.append([np.nan] * len(desc_names))
            desc_valid.append(False)

    desc_df = pd.DataFrame(desc_records, columns=desc_names)
    print(f"2D descriptors done. Valid molecules: {sum(desc_valid)} / {len(smiles_list)}")

    ecfp_col_names = [f'ecfp_{i}' for i in range(ECFP_BITS)]
    ecfp_records = []
    ecfp_valid = []

    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
            ecfp_records.append(calculate_fingerprint(mol, fp_type="ecfp4"))
            ecfp_valid.append(True)
        except Exception:
            ecfp_records.append([np.nan] * ECFP_BITS)
            ecfp_valid.append(False)

    ecfp_df = pd.DataFrame(ecfp_records, columns=ecfp_col_names)
    print(f"ECFP4 fingerprints done. Valid molecules: {sum(ecfp_valid)} / {len(smiles_list)}")

    maccs_col_names = [f'maccs_{i}' for i in range(MACCS_BITS)]
    maccs_records = []
    maccs_valid = []

    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
            maccs_records.append(calculate_fingerprint(mol, fp_type="maccs"))
            maccs_valid.append(True)
        except Exception:
            maccs_records.append([np.nan] * MACCS_BITS)
            maccs_valid.append(False)

    maccs_df = pd.DataFrame(maccs_records, columns=maccs_col_names)
    print(f"MACCS fingerprints done. Valid molecules: {sum(maccs_valid)} / {len(smiles_list)}")

    full_df = pd.concat(
        [
            pd.Series(smiles_list, name='smiles'),
            desc_df,
            ecfp_df,
            maccs_df
        ],
        axis=1
    )
    full_df['is_valid'] = np.logical_and.reduce([desc_valid, ecfp_valid, maccs_valid])
    clean_df = full_df[full_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

    clean_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("Feature calculation completed")
    print(f"Original molecules: {len(smiles_list)}")
    print(f"Valid molecules (all features): {len(clean_df)}")
    print(f"Filtered invalid: {len(smiles_list) - len(clean_df)}")
    print(f"Total feature dimensions (excl. smiles): {clean_df.shape[1] - 1}")
    print(f"  - 2D descriptors: {len(desc_names)}")
    print(f"  - ECFP4: {ECFP_BITS}")
    print(f"  - MACCS: {MACCS_BITS}")
    print(f"Output saved: {OUTPUT_CSV}")
