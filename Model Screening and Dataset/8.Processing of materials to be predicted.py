import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Composition, Element
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial import cKDTree
from tqdm import tqdm

API_KEY = 'suMUrr8V7GHweOAozjDrOH3rRqG8RmFX'
INPUT_FILE = 'mp_export_final.csv'
OUTPUT_FILE = 'materials_to_predict.csv'

PROBE_R = 1.2
FINE_SPACING = 0.3
TH_APF = 0.70
TH_RMAX = 2.0


def get_robust_radius(specie_or_el):
    """Logic for obtaining atomic radius """
    try:
        el = Element(specie_or_el) if isinstance(specie_or_el, str) else specie_or_el
        return el.vdw_radius or el.atomic_radius or el.covalent_radius or 1.5
    except:
        return 1.5


def calc_porosity_logic(structure, spacing=0.3):
    """Core algorithm for porosity """
    if not structure: return 0.0, 0.0
    l = structure.lattice
    nx, ny, nz = [max(5, int(x / spacing)) for x in l.abc]
    grid_size = nx * ny * nz
    x, y, z = np.linspace(0, 1, nx, endpoint=False), np.linspace(0, 1, ny, endpoint=False), np.linspace(0, 1, nz,
                                                                                                        endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cart_points = l.get_cartesian_coords(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    atom_coords, atom_radii = structure.cart_coords, np.array([get_robust_radius(s.specie) for s in structure])
    tree = cKDTree(atom_coords)
    dists, _ = tree.query(cart_points, k=1)

    # Total porosity vs. accessible porosity
    total_p = max(1.0 - (np.sum(dists < atom_radii) / grid_size), 0.0)
    acc_p = max(1.0 - (np.sum(dists < (atom_radii + PROBE_R)) / grid_size), 0.0)
    return total_p, acc_p


def run_pipeline():
    df_base = pd.read_csv(INPUT_FILE)
    m_ids = df_base['material_id'].unique().tolist()
    print(f"deal {len(m_ids)} data...")

    results = []
    batch_size = 50

    with MPRester(API_KEY) as mpr:
        for i in range(0, len(m_ids), batch_size):
            batch = m_ids[i:i + batch_size]
            try:
                docs = mpr.materials.summary.search(material_ids=batch, fields=['material_id', 'structure'])
                struct_map = {str(d.material_id): d.structure for d in docs if d.structure}
            except:
                continue

            current_df = df_base[df_base['material_id'].isin(batch)]
            for row in tqdm(current_df.to_dict(orient='records'), desc=f"Batch {i // batch_size + 1}"):
                mid = str(row['material_id'])
                res = row.copy()

                try:
                    comp = Composition(row['formula'])
                    eneg = [Element(str(el)).X for el in comp.elements if Element(str(el)).X]
                    radii = [Element(str(el)).atomic_radius for el in comp.elements if Element(str(el)).atomic_radius]
                    res.update({
                        'Electronegativity difference': round(max(eneg) - min(eneg), 3) if eneg else np.nan,
                        'Average electronegativity': round(np.mean(eneg), 3) if eneg else np.nan,
                        'Average atomic radius(Å)': round(np.mean(radii), 3) if radii else np.nan,
                        ' F element atomic fraction': round(comp.get_atomic_fraction(Element('F')), 3),
                        ' C element atomic fraction': round(comp.get_atomic_fraction(Element('C')), 3)
                    })
                except:
                    pass

                s = struct_map.get(mid)
                if s:
                    try:
                        nn = CrystalNN()
                        all_cn, bond_lengths, covalent_count = [], [], 0
                        for idx, site in enumerate(s):
                            neighs = nn.get_nn_info(s, idx)
                            all_cn.append(len(neighs))
                            for n in neighs:
                                bl = site.distance(n['site'])
                                bond_lengths.append(bl)
                                if abs(Element(str(site.specie)).X - Element(str(n['site'].specie)).X) < 1.7:
                                    covalent_count += 1

                        vol_atoms = sum((4 / 3) * np.pi * (get_robust_radius(site.specie) ** 3) for site in s)
                        apf = min(vol_atoms / s.volume, 1.0)

                        # r_max
                        l = s.lattice
                        grid = np.column_stack(
                            np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10),
                                        indexing='ij')).reshape(-1, 3)
                        tree = cKDTree(s.cart_coords)
                        dists, _ = tree.query(l.get_cartesian_coords(grid), k=1)
                        rmax = np.max(dists - 1.5)  # 简化版快搜

                        total_p, acc_p = 0.0, 0.0
                        if apf < TH_APF and rmax > TH_RMAX:
                            total_p, acc_p = calc_porosity_logic(s, spacing=FINE_SPACING)

                        res.update({
                            'Average bond length(Å)': round(np.mean(bond_lengths), 3) if bond_lengths else np.nan,
                            'Covalent bond ratio (estimated)': round(covalent_count / len(bond_lengths), 3) if bond_lengths else 0,
                            'Average coordination number(CN)': round(np.mean(all_cn), 3) if all_cn else np.nan,
                            'APF': round(apf, 3),
                            'r_max': round(rmax, 3),
                            'porosity_total': round(total_p, 4),
                            'porosity_acc': round(acc_p, 4),
                            'is_porous': 1 if rmax > TH_RMAX else 0
                        })
                    except:
                        pass

                results.append(res)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"generated.: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()
