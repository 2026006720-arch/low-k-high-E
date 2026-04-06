import pandas as pd
import numpy as np

# Step 1. Read dataset
filename = 'materials_to_predict.csv'
df = pd.read_csv(filename)
print(f'read data，totally detected {len(df.columns)} original feature columns ')

# --- Dynamically retrieve the screening list ---
selected_features_file = 'materials_selected_features.csv'
# Read only the header and use the list for loop design.
target_columns_raw = pd.read_csv(selected_features_file, nrows=0).columns.tolist()

# Iterate, excluding data_source.
target_columns = [col for col in target_columns_raw if col != 'data_source']
print(f'Reference feature list extraction completed, target retained feature count: {len(target_columns)}')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['material_id', 'formula', 'elements']
base_features = [c for c in numeric_cols if c not in exclude_cols]

X = df[base_features].copy()

# Step 2. Feature expansion
print("Feature expanding...")
X_expanded = X.copy()

col_porosity = 'porosity_total'
col_porosity_acc = 'porosity_acc'

# --- A. Dielectric physical mechanism ---
X_expanded["penn_proxy"] = X["density (g/cm³)"] / (X["band_gap (eV)"] ** 2 + 0.1)
X_expanded["polarizability_proxy"] = X["Average atomic radius(Å)"] / (X["Average electronegativity"] + 0.1)
X_expanded["ionic_polarizability_proxy"] = X["Average atomic radius(Å)"] * X["Electronegativity difference"]
# Dielectric performance
X_expanded["dielectric_efficiency"] = X[col_porosity] * X["band_gap (eV)"]
X_expanded["electron_density_proxy"] = X["density (g/cm³)"] / (X["Average atomic radius(Å)"] ** 3 + 0.1)
X_expanded["electron_ion"] = X["band_gap (eV)"] * X["Average electronegativity"]

# --- B. Mechanical-structural trade-off ---
X_expanded["specific_G"] = X["shear_modulus_G_VRH (GPa)"] / (X["density (g/cm³)"] + 0.1)
X_expanded["specific_K"] = X["bulk_modulus_K_VRH (GPa)"] / (X["density (g/cm³)"] + 0.1)
X_expanded["brittleness_index"] = X["shear_modulus_G_VRH (GPa)"] / (X["bulk_modulus_K_VRH (GPa)"] + 1.0)

# --- C. Chemical bonding features ---
X_expanded["ionicity_index"] = X["Electronegativity difference"] ** 2
X_expanded["fluorine_weighted"] = X[" F element atomic fraction"] * X["Average electronegativity"]
X_expanded["carbon_weighted"] = X[" C element atomic fraction"] * X["Average electronegativity"]
X_expanded["covalent_coordination_proxy"] = X["Covalent bond ratio (estimated)"] * X["Average coordination number(CN)"]

# --- D. Structural compactness ---
X_expanded["bond_gap_product"] = X["Average bond length(Å)"] * X["band_gap (eV)"]
X_expanded["effective_density"] = X["density (g/cm³)"] * (1 - X[col_porosity])
X_expanded["void_size_ratio"] = X[col_porosity] / (X["r_max"] + 0.1)

# --- E. Derived cross features ---
interaction_pairs = [
    (col_porosity, "band_gap (eV)"),
    ("density (g/cm³)", "band_gap (eV)"),
    (col_porosity, "density (g/cm³)"),
    (col_porosity, "shear_modulus_G_VRH (GPa)"),
    (col_porosity, "bulk_modulus_K_VRH (GPa)"),
    ("Average electronegativity", "Average atomic radius(Å)"),
    ("epsilon_ratio", "APF")
]
for a, b in interaction_pairs:
    if a in X.columns and b in X.columns:
        X_expanded[f"{a}_x_{b}"] = X[a] * X[b]

# --- F. Derived cross features ---
math_cols = ["density (g/cm³)", "band_gap (eV)", "epsilon_ratio", "porosity_total", "APF"]
for col in math_cols:
    if col in X.columns:
        X_expanded[f"log_{col}"] = np.log(X[col] + 0.01)
        X_expanded[f"{col}_squared"] = X[col] ** 2

X_expanded = X_expanded.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Feature expansion completed！")

# Step 3. Filter and save
# Merge the identifier column with all processed features.
temp_df = pd.concat([df[['material_id', 'formula']], X_expanded], axis=1)

# Iterate to keep only the columns that are both in the target_columns list and present in temp_df.
final_cols = [c for c in target_columns if c in temp_df.columns]
df_final = temp_df[final_cols]

# Save to a new file.
output_filename = 'materials_selected_features2.csv'
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Done！Final file: {output_filename}")
print(f"Final number of retained feature columns: {df_final.shape[1]}")



