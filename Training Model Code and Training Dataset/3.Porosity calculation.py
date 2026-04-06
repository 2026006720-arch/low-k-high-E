import pandas as pd
import numpy as np
import time
from mp_api.client import MPRester
from pymatgen.core import Element
from scipy.spatial import cKDTree
from tqdm import tqdm

INPUT_FILE = 'materials_features1.csv'
API_KEY = 'suMUrr8V7GHweOAozjDrOH3rRqG8RmFX'

# Keep the threshold unchanged, or adjust as needed.
TH_APF = 0.70
TH_RMAX = 2.0
PROBE_R = 1.2  # Probe radius
FINE_SPACING = 0.3


# ================= Calculation function =================

def get_robust_radius(specie_or_el):
    """Keep the original radius priority logic unchanged."""
    try:
        if isinstance(specie_or_el, str):
            el = Element(specie_or_el)
        elif hasattr(specie_or_el, 'element'):
            el = specie_or_el.element
        else:
            el = specie_or_el

        if el.vdw_radius is not None: return el.vdw_radius
        if el.atomic_radius is not None: return el.atomic_radius
        if el.covalent_radius is not None: return el.covalent_radius
        return 1.5
    except:
        return 1.5


def calc_porosity_types(structure, probe_r=1.2, spacing=0.3):
    """
    Refine the step size and simultaneously calculate the total porosity and connected porosity.
    """
    if not structure: return 0.0, 0.0

    l = structure.lattice
    # Refine grid step size
    nx, ny, nz = [max(5, int(x / spacing)) for x in l.abc]
    grid_size = nx * ny * nz

    # Generate grid points
    x, y, z = np.linspace(0, 1, nx, endpoint=False), np.linspace(0, 1, ny, endpoint=False), np.linspace(0, 1, nz,
                                                                                                        endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cart_points = l.get_cartesian_coords(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))

    atom_coords = structure.cart_coords
    atom_radii = np.array([get_robust_radius(s.specie) for s in structure])

    tree = cKDTree(atom_coords)
    dists, idxs = tree.query(cart_points, k=1)

    # 1. Calculate total porosity: A point is counted as a pore if it is not within any atomic radius.
    occupied_total = dists < atom_radii[idxs]
    total_porosity = max(1.0 - (np.sum(occupied_total) / grid_size), 0.0)

    # 2. Calculate accessible porosity: take the probe radius into account.
    # Logic: The distance from the point to the atom must be greater than (atomic radius + probe radius) to be considered accessible.
    atom_radii_probe = atom_radii + probe_r
    occupied_acc = dists < atom_radii_probe[idxs]
    acc_porosity = max(1.0 - (np.sum(occupied_acc) / grid_size), 0.0)

    return total_porosity, acc_porosity


def calc_apf(structure):
    """Remain unchanged"""
    if not structure: return np.nan
    vol_atoms = 0.0
    for site in structure:
        r = get_robust_radius(site.specie)
        vol_atoms += (4 / 3) * np.pi * (r ** 3)
    return min(vol_atoms / structure.volume, 1.0)


def calc_r_max_fast(structure, spacing=0.5):
    """Slightly improve the detection accuracy of r_max."""
    if not structure: return 0.0
    l = structure.lattice
    nx, ny, nz = [max(3, int(x / spacing)) for x in l.abc]
    x, y, z = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cart_points = l.get_cartesian_coords(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))

    atom_coords = structure.cart_coords
    atom_radii = np.array([get_robust_radius(s.specie) for s in structure])

    tree = cKDTree(atom_coords)
    dists, idxs = tree.query(cart_points, k=1)
    val = np.max(dists - atom_radii[idxs])
    return val if val > 0 else 0.0


# ================= Main workflow =================

def main_process():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"load {len(df)} data...")
    except FileNotFoundError:
        print("Error: File not found")
        return None

    m_ids = df['material_id'].tolist()
    struct_map = {}

    print("Step 1/2: Downloading crystal structure....")
    with MPRester(API_KEY) as mpr:
        batch_size = 250
        for i in tqdm(range(0, len(m_ids), batch_size)):
            try:
                batch_ids = m_ids[i:i + batch_size]
                docs = mpr.materials.summary.search(material_ids=batch_ids, fields=["material_id", "structure"])
                for d in docs:
                    struct_map[d.material_id] = d.structure
            except Exception as e:
                print(f"Error downloading batch {i} : {e}")
                pass

    results = []
    print(f"Step 2/2: Calculate refined features (Spacing={FINE_SPACING})...")

    for mid in tqdm(m_ids):
        s = struct_map.get(mid)
        # Initialize newly added feature fields.
        apf, r_max, is_porous = np.nan, np.nan, 0
        total_porosity, acc_porosity = 0.0, 0.0

        if s:
            try:
                apf = calc_apf(s)
                r_max = calc_r_max_fast(s)

                # Filter condition
                if (apf < TH_APF) and (r_max > TH_RMAX):
                    is_porous = 1
                    # Call the new calculation function.
                    total_porosity, acc_porosity = calc_porosity_types(s, probe_r=PROBE_R, spacing=FINE_SPACING)
            except Exception as e:
                # Silently handle errors to avoid interruption.
                pass

        # Add new features to the result list.
        results.append([mid, apf, r_max, is_porous, total_porosity, acc_porosity])

    df_res = pd.DataFrame(results,
                          columns=['material_id', 'APF', 'r_max', 'is_porous', 'porosity_total', 'porosity_acc'])

    df_final = pd.merge(df, df_res, on='material_id', how='left')
    return df_final


if __name__ == "__main__":
    final_df = main_process()
    if final_df is not None:
        output_name = 'materials_features2.csv'
        final_df.to_csv(output_name, index=False)
        print("\n" + "=" * 40)
        print(f" Refined calculation completed.！")
        print(f"New field: [porosity_total] (Total porosity), [porosity_acc] (Connected/accessible porosity)")
        print(f"Current sampling step size: {FINE_SPACING} Å")
        print(f"Save results to: {output_name}")
        print("=" * 40)
