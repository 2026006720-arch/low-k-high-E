import pandas as pd
import numpy as np
from tqdm import tqdm
from mp_api.client import MPRester

# API Key
API_KEY = 'suMUrr8V7GHweOAozjDrOH3rRqG8RmFX'

def compute_KG_from_Cij(C):
    """Calculate K and G from the elastic matrix, and incorporate the Born stability criterion. """
    try:
        C = np.array(C, dtype=float)
        if C.shape != (6, 6): return (np.nan, np.nan)
        eigenvalues = np.linalg.eigvals(C)
        if np.any(eigenvalues <= 0): return (np.nan, np.nan)

        C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
        C12, C13, C23 = C[0, 1], C[0, 2], C[1, 2]
        C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]

        K_V = (C11 + C22 + C33 + 2 * (C12 + C13 + C23)) / 9.0
        G_V = (C11 + C22 + C33 - (C12 + C13 + C23) + 3 * (C44 + C55 + C66)) / 15.0

        S = np.linalg.inv(C)
        S11, S22, S33 = S[0, 0], S[1, 1], S[2, 2]
        S12, S13, S23 = S[0, 1], S[0, 2], S[1, 2]
        S44, S55, S66 = S[3, 3], S[4, 4], S[5, 5]

        denom_KR = (S11 + S22 + S33 + 2 * (S12 + S13 + S23))
        denom_GR = (4 * (S11 + S22 + S33) - 4 * (S12 + S13 + S23) + 3 * (S44 + S55 + S66))

        K_R = 1.0 / denom_KR if denom_KR != 0 else np.nan
        G_R = 15.0 / denom_GR if denom_GR != 0 else np.nan

        final_K = np.nanmean([K_V, K_R])
        final_G = np.nanmean([G_V, G_R])
        return (final_K, final_G) if final_K > 0 and final_G > 0 else (np.nan, np.nan)
    except:
        return (np.nan, np.nan)

def get_dielectric_features(tensor):
    """
    Extract tensor features: mean, variance, maximum range, anisotropy ratio.
    """
    try:
        arr = np.array(tensor, dtype=float)
        if arr.shape == (3, 3):
            # Get eigenvalues and sort them.
            eigvals = np.linalg.eigvals(arr)
            eigvals = np.sort(eigvals)

            mean_val = float(np.mean(eigvals))
            var_val = float(np.var(eigvals))  # variance
            delta_val = float(eigvals[-1] - eigvals[0])  # range
            ratio_val = float(eigvals[-1] / eigvals[0]) if eigvals[0] != 0 else np.nan

            return mean_val, var_val, delta_val, ratio_val
    except:
        pass
    return np.nan, np.nan, np.nan, np.nan

def run_export():
    with MPRester(API_KEY) as mpr:
        excluded = ['Li', 'Fe', 'Na', 'Zn', 'K', 'Al', 'Cu', 'Ni', 'Pb', 'Cd', 'Hg', 'As', 'Tl', 'Be','Lu','Re',
            'Au', 'Pt', 'Ir', 'Os', 'Pd', 'Ru', 'Rh', 'Tc', 'Pm', 'Po', 'At', 'Co', 'Se', 'Sb','Bi','Yb','Re',
            'Rn', 'Fr', 'Ra', 'Ac', 'Lr', 'U', 'Th', 'Pu', 'V', 'Cr', 'Mn', 'He', 'Ne', 'Ar','Y','Rb','Mo',
            'Kr', 'Xe', 'Ag', 'Ba','F','Sn','Sm','Dy','Pr','Ho','Nd','Gd','Tm','Er','Tb','Gd','Te','Tm','Cs','Sr',
            'Ga','In','Ge','W','Nb','Sc','F','Cl','Br','I','Np','Pa',
        ]

        print("1. Getting initial screening list....")
        raw_docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.1),
            is_stable=True,
            chunk_size=1000,
            fields=[
                'material_id', 'formula_pretty', 'energy_above_hull',
                'elements', 'band_gap', 'density', 'energy_per_atom'
            ]
        )

        docs = []
        for doc in raw_docs:
            mat_elements = [str(el) for el in doc.elements]
            if not any(el in excluded for el in mat_elements):
                docs.append(doc)

        print(f"Number of remaining materials after excluding specific elements: {len(docs)}")

        mpids = [str(doc.material_id) for doc in docs]
        all_results = []
        batch_size = 500

        for i in range(0, len(mpids), batch_size):
            batch_ids = mpids[i:i + batch_size]
            diel_data = mpr.materials.dielectric.search(material_ids=batch_ids)
            elas_data = mpr.materials.elasticity.search(material_ids=batch_ids)

            diel_dict = {str(d.material_id): d for d in diel_data}
            elas_dict = {str(e.material_id): e for e in elas_data}

            for mat in tqdm(docs[i:i + batch_size], desc=f"处理批次 {i // batch_size + 1}"):
                mid = str(mat.material_id)

                # --- Dielectric feature acquisition ---
                diel = diel_dict.get(mid)
                k_val = k_var = k_delta = k_ratio = np.nan
                if diel:
                    t = getattr(diel, 'electronic', None) or getattr(diel, 'total', None) or getattr(diel, 'e_electronic', None)
                    if t is not None:
                        k_val, k_var, k_delta, k_ratio = get_dielectric_features(t)

                # --- Elastic modulus extraction ---
                elas = elas_dict.get(mid)
                K = G = E = np.nan
                if elas:
                    c_ij = getattr(elas, 'elastic_tensor', None)
                    if c_ij is not None:
                        raw_matrix = getattr(c_ij, 'ieee_format', c_ij)
                        K, G = compute_KG_from_Cij(raw_matrix)

                if not np.isnan(K) and not np.isnan(G):
                    E = 9 * K * G / (3 * K + G)

                # --- Save results ---
                all_results.append({
                    "material_id": mid,
                    "formula": mat.formula_pretty,
                    "energy_above_hull (eV/atom)": mat.energy_above_hull,
                    "energy_per_atom (eV/atom)": mat.energy_per_atom,
                    "density (g/cm³)": mat.density,
                    "band_gap (eV)": mat.band_gap,
                    "bulk_modulus_K_VRH (GPa)": K,
                    "shear_modulus_G_VRH (GPa)": G,
                    "youngs_modulus_E(GPa)": E,
                    "epsilon(k)": k_val if not np.isnan(k_val) else None,
                    "epsilon_var": k_var if not np.isnan(k_var) else None,
                    "epsilon_delta": k_delta if not np.isnan(k_delta) else None,
                    "epsilon_ratio": k_ratio if not np.isnan(k_ratio) else None
                })

        df = pd.DataFrame(all_results)
        df.to_csv("mp_export_final.csv", index=False, encoding='utf-8-sig')
        print(f"Export completed! The final generated file contains {len(df)} material records.")

if __name__ == "__main__":
    run_export()


