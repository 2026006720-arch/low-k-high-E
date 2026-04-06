import numpy as np
import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.core import Element
from scipy.stats import zscore

# API Key
API_KEY = 'suMUrr8V7GHweOAozjDrOH3rRqG8RmFX'


# utility function
def extract_numeric(value):
    if value is None:
        return np.nan
    elif isinstance(value, float):
        return value
    try:
        return value.to('1')
    except:
        return np.nan


def numeric_from_field(x, prefer_keys=None):
    import numpy as _np
    if x is None: return _np.nan
    try:
        if hasattr(x, 'dict'): x = x.dict()
    except:
        pass
    if isinstance(x, (int, float, _np.floating, _np.integer)): return float(x)
    if isinstance(x, dict):
        if prefer_keys:
            for k in prefer_keys:
                if k in x and isinstance(x[k], (int, float)): return float(x[k])
        for k in ('K_VRH', 'G_VRH', 'K_Voigt', 'G_Voigt', 'K_Reuss', 'G_Reuss', 'value', 'total'):
            if k in x and isinstance(x[k], (int, float)): return float(x[k])
        nums = [v for v in x.values() if isinstance(v, (int, float))]
        if nums: return float(sum(nums) / len(nums))
        for v in x.values():
            val = numeric_from_field(v, prefer_keys)
            if not _np.isnan(val): return val
        return _np.nan
    if isinstance(x, (list, tuple)):
        nums = [v for v in x if isinstance(v, (int, float))]
        if nums: return float(sum(nums) / len(nums))
        for v in x:
            val = numeric_from_field(v, prefer_keys)
            if not _np.isnan(val): return val
        return _np.nan
    return _np.nan


def compute_KG_from_Cij(C):
    """
    Calculate the bulk modulus K and shear modulus G from the elastic matrix,
    and include the Born stability criterion (positive definiteness check of the matrix).
    """
    try:
        C = np.array(C, dtype=float)
        if C.shape != (6, 6):
            return (np.nan, np.nan)

        eigenvalues = np.linalg.eigvals(C)
        if np.any(eigenvalues <= 0):
            return (np.nan, np.nan)

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

        if final_K <= 0 or final_G <= 0:
            return (np.nan, np.nan)

        return (final_K, final_G)
    except:
        return (np.nan, np.nan)


# --- anisotropic feature extraction function ---
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


# Data fetching
all_data = []
batch_size = 1000
with MPRester(API_KEY) as mpr:
    print('Searching materials with dielectric data...')
    docs = mpr.materials.summary.search(
        has_props=['dielectric'],
        fields=['material_id', 'formula_pretty', 'density', 'band_gap', 'energy_per_atom', 'energy_above_hull',
                'elements']
    )
    mpids = [doc.material_id for doc in docs]
    print(f'Found {len(mpids)} materials with dielectric info')

    for i in range(0, len(mpids), batch_size):
        batch_ids = mpids[i:i + batch_size]
        diel_data = mpr.materials.dielectric.search(material_ids=batch_ids)
        elas_data = mpr.materials.elasticity.search(batch_ids, fields=['material_id', 'bulk_modulus', 'shear_modulus',
                                                                       'elastic_tensor'])

        diel_dict = {getattr(d, 'material_id', None): d for d in diel_data}
        elas_dict = {getattr(e, 'material_id', None): e for e in elas_data}

        for mat in tqdm(docs[i:i + batch_size], desc=f"Processing batch {i // batch_size + 1}"):
            mid = mat.material_id
            formula = mat.formula_pretty
            density = mat.density or np.nan
            band_gap = mat.band_gap or np.nan
            energy_per_atom = mat.energy_per_atom or np.nan
            energy_above_hull = getattr(mat, 'energy_above_hull', np.nan)
            elements = [str(e) for e in mat.elements]

            # --- Dielectric constant and anisotropy treatment ---
            diel = diel_dict.get(mid, None)
            k_val = k_var = k_delta = k_ratio = np.nan

            if diel:
                tensor_to_use = None
                if hasattr(diel, 'electronic') and diel.electronic is not None:
                    tensor_to_use = diel.electronic
                elif hasattr(diel, 'total') and diel.total is not None:
                    tensor_to_use = diel.total
                elif hasattr(diel, 'e_electronic') and diel.e_electronic is not None:
                    tensor_to_use = diel.e_electronic

                if tensor_to_use is not None:
                    k_val, k_var, k_delta, k_ratio = get_dielectric_features(tensor_to_use)

            # Skip if the mean is invalid.
            if np.isnan(k_val):
                continue

            # --- elastic modulus part ---
            elas = elas_dict.get(mid, None)
            K = G = E = np.nan

            if elas:
                if hasattr(elas, 'elastic_tensor') and elas.elastic_tensor is not None:
                    K, G = compute_KG_from_Cij(elas.elastic_tensor)
                if np.isnan(K) or np.isnan(G):
                    K = numeric_from_field(getattr(elas, 'bulk_modulus', None), prefer_keys=['K_VRH'])
                    G = numeric_from_field(getattr(elas, 'shear_modulus', None), prefer_keys=['G_VRH'])
                if not np.isnan(K) and not np.isnan(G) and K > 0 and G > 0:
                    if (3 * K + G) != 0:
                        E = 9 * K * G / (3 * K + G)

            all_data.append([mid, formula, density, band_gap, energy_per_atom, energy_above_hull,
                             k_val, k_var, k_delta, k_ratio, K, G, E, elements])

# Data saving and cleaning
df_raw = pd.DataFrame(all_data, columns=[
    'material_id', 'formula', 'density (g/cm³)', 'band_gap (eV)', 'energy_per_atom (eV/atom)',
    'energy_above_hull (eV/atom)', 'epsilon(k)', 'epsilon_var', 'epsilon_delta', 'epsilon_ratio',
    'bulk_modulus_K_VRH (GPa)', 'shear_modulus_G_VRH (GPa)', 'youngs_modulus_E(GPa)', 'elements'])

df_raw.to_csv('materials_raw.csv', index=False)

df = df_raw.copy()
df = df[df['epsilon(k)'] < 4.0]
df = df[df['energy_above_hull (eV/atom)'] < 0.1]

# Clean outliers using the mean
z_scores = np.abs(zscore(df['epsilon(k)']))
df = df[z_scores < 3]

df.to_csv('materials_clean.csv', index=False)
print(f'Clean data size: {len(df)}')
