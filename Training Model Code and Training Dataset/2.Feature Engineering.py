import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Composition, Element
from pymatgen.analysis.local_env import CrystalNN
from tqdm import tqdm

API_KEY = 'suMUrr8V7GHweOAozjDrOH3rRqG8RmFX'
INPUT_FILE = 'materials_clean.csv'

# ================= Step 1: Read data =================
df_base = pd.read_csv(INPUT_FILE)
print(f'read {len(df_base)} data')

id_col = next((c for c in df_base.columns if 'id' in c.lower()), None)
if id_col is None:
    raise KeyError("material_id column not found. Please confirm that the file contains 'material_id'")

material_ids = df_base[id_col].dropna().unique().tolist()
print(f'finally find {len(material_ids)} material_id')


# ================= utility function：batch =================
def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ================= Step 2: Feature extraction =================
results = []

with MPRester(API_KEY) as mpr:
    # Convert to a list so that tqdm can display the progress bar.
    batches = list(batch_list(material_ids, 50))
    for batch in tqdm(batches, desc='Processing batches'):
        try:
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=['material_id', 'structure']
            )
        except Exception as e:
            print(f'Batch acquisition failed: {e}')
            continue

        # Current batch structure cache
        struct_map = {
            d.material_id: d.structure
            for d in docs
            if d.structure is not None
        }

        batch_set = set(batch)

        # Only iterate over the materials in the current batch, greatly improving efficiency.
        # Use loc and isin to filter the rows of the DataFrame for the current batch.
        current_batch_df = df_base[df_base[id_col].isin(batch_set)]

        for d in current_batch_df.to_dict(orient='records'):
            mid = d.get(id_col)
            if mid is None:
                continue

            # ------------------ Chemical features ------------------
            try:
                comp = Composition(d['formula'])

                eneg = [
                    Element(str(el)).X
                    for el in comp.elements
                    if Element(str(el)).X is not None
                ]
                radius = [
                    Element(str(el)).atomic_radius
                    for el in comp.elements
                    if Element(str(el)).atomic_radius is not None
                ]

                eneg_diff = max(eneg) - min(eneg) if eneg else np.nan
                avg_eneg = np.mean(eneg) if eneg else np.nan
                avg_radius = np.mean(radius) if radius else np.nan

                frac_F = comp.get_atomic_fraction(Element('F')) if 'F' in [str(el) for el in comp.elements] else 0
                frac_C = comp.get_atomic_fraction(Element('C')) if 'C' in [str(el) for el in comp.elements] else 0

            except Exception:
                eneg_diff = avg_eneg = avg_radius = frac_F = frac_C = np.nan

            # ------------------ Structural features ------------------
            avg_bond_length = covalent_bond_ratio = avg_cn = np.nan

            try:
                s = struct_map.get(mid)
                if s is None:
                    # If the API does not return a structure, skip the structure calculation.
                    results.append({
                        'material_id': mid,
                        'Electronegativity difference': round(eneg_diff, 3) if pd.notna(eneg_diff) else np.nan,
                        'Average electronegativity': round(avg_eneg, 3) if pd.notna(avg_eneg) else np.nan,
                        'Average atomic radius(Å)': round(avg_radius, 3) if pd.notna(avg_radius) else np.nan,
                        ' F element atomic fraction': round(frac_F, 3),
                        ' C element atomic fraction': round(frac_C, 3),
                        'Average bond length(Å)': np.nan,
                        'Covalent bond ratio (estimated)': np.nan,
                        'Average coordination number(CN)': np.nan,
                    })
                    continue

                nn = CrystalNN()
                bond_lengths = []
                covalent_count = 0
                total_bonds = 0
                all_cn = []

                for i, site in enumerate(s):
                    # Get coordination information
                    neighs = nn.get_nn_info(s, i)
                    # 1. Core feature: record the coordination number of this atom.
                    all_cn.append(len(neighs))

                    for n in neighs:
                        # 2. Calculate bond length.
                        bond_length = site.distance(n['site'])
                        bond_lengths.append(bond_length)
                        total_bonds += 1

                        # 3. Calculate covalent bond ratio.
                        el1, el2 = site.specie, n['site'].specie
                        X1 = Element(str(el1)).X
                        X2 = Element(str(el2)).X
                        if X1 is not None and X2 is not None:
                            # If the electronegativity difference is less than 1.7, classify as covalent/metallic tendency.
                            if abs(X1 - X2) < 1.7:
                                covalent_count += 1

                if bond_lengths:
                    avg_bond_length = np.mean(bond_lengths)
                    covalent_bond_ratio = (
                        covalent_count / total_bonds if total_bonds > 0 else np.nan
                    )

                if all_cn:
                    # Average coordination number of all atoms in this material.
                    avg_cn = np.mean(all_cn)

            except Exception as e:
                print(f'{mid} 结构处理失败:', e)

            # ------------------ Save results ------------------
            results.append({
                'material_id': mid,
                'Electronegativity difference': round(eneg_diff, 3) if pd.notna(eneg_diff) else np.nan,
                'Average electronegativity': round(avg_eneg, 3) if pd.notna(avg_eneg) else np.nan,
                'Average atomic radius(Å)': round(avg_radius, 3) if pd.notna(avg_radius) else np.nan,
                ' F element atomic fraction': round(frac_F, 3),
                ' C element atomic fraction': round(frac_C, 3),
                'Average bond length(Å)': round(avg_bond_length, 3) if pd.notna(avg_bond_length) else np.nan,
                'Covalent bond ratio (estimated)': round(covalent_bond_ratio, 3) if pd.notna(covalent_bond_ratio) else np.nan,
                'Average coordination number(CN)': round(avg_cn, 3) if pd.notna(avg_cn) else np.nan,
            })

# ================= Step 3: Export =================
df_features1 = pd.DataFrame(results).drop_duplicates(subset=['material_id'])
df_features1.to_csv('materials_features1.csv', index=False, encoding='utf-8-sig')

print(f'Step1 finish，save {len(df_features1)} data')

