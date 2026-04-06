import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Step 1. Read the dataset containing all new features.
# ==========================================
filename = 'materials_features_filled.csv'
df = pd.read_csv(filename)
print(f'Data read successfully,totally detected {len(df.columns)} original feature columns')

# Target variable
target_k = 'epsilon(k)'
target_E = 'youngs_modulus_E(GPa)'

# Automatically extract all numerical features and exclude non-feature columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['material_id', 'formula', 'elements', 'data_source', target_k, target_E]
base_features = [c for c in numeric_cols if c not in exclude_cols]

X = df[base_features].copy()

# ==========================================
# Step 2. Feature expansion (incorporating new features)
# ==========================================
print("Feature expansion...")
X_expanded = X.copy()

# Lock the key new column names.
col_porosity = 'porosity_total'
col_porosity_acc = 'porosity_acc'

# --- A. Dielectric physical mechanism (restoring formulas and introducing anisotropy) ---
X_expanded["penn_proxy"] = X["density (g/cm³)"] / (X["band_gap (eV)"] ** 2 + 0.1)
X_expanded["polarizability_proxy"] = X["Average atomic radius(Å)"] / (X["Average electronegativity"] + 0.1)
X_expanded["ionic_polarizability_proxy"] = X["Average atomic radius(Å)"] * X["Electronegativity difference"]
# Dielectric performance
X_expanded["dielectric_efficiency"] = X[col_porosity] * X["band_gap (eV)"]
X_expanded["electron_density_proxy"] = X["density (g/cm³)"] / (X["Average atomic radius(Å)"] ** 3 + 0.1)
X_expanded["electron_ion"] = X["band_gap (eV)"] * X["Average electronegativity"]

# --- B. Mechanical-structural trade-off (using imputed K/G/E) ---
X_expanded["specific_G"] = X["shear_modulus_G_VRH (GPa)"] / (X["density (g/cm³)"] + 0.1)
X_expanded["specific_K"] = X["bulk_modulus_K_VRH (GPa)"] / (X["density (g/cm³)"] + 0.1)
X_expanded["brittleness_index"] = X["shear_modulus_G_VRH (GPa)"] / (X["bulk_modulus_K_VRH (GPa)"] + 1.0)

# --- C. Chemical bonding features (incorporating the newly generated covalent bond ratio) ---
X_expanded["ionicity_index"] = X["Electronegativity difference"] ** 2
X_expanded["fluorine_weighted"] = X[" F element atomic fraction"] * X["Average electronegativity"]
X_expanded["carbon_weighted"] = X[" C element atomic fraction"] * X["Average electronegativity"]
X_expanded["covalent_coordination_proxy"] = X["Covalent bond ratio (estimated)"] * X["Average coordination number(CN)"]

# --- D. Structural compactness (incorporating APF and r_max) ---
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

# --- F. Mathematical transformation of new features  ---
math_cols = ["density (g/cm³)", "band_gap (eV)", "epsilon_ratio", "porosity_total", "APF"]
for col in math_cols:
    if col in X.columns:
        X_expanded[f"log_{col}"] = np.log(X[col] + 0.01)
        X_expanded[f"{col}_squared"] = X[col] ** 2

X_expanded = X_expanded.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Total number of features reaches: {X_expanded.shape[1]}")

# ==========================================
# Step 3. RFE selection (8/2 split to prevent leakage)
# ==========================================
print("\nStart executing the RFE selection model...")

real_mask = df['data_source'] == 0
X_real = X_expanded.loc[real_mask]
y_k_real = df.loc[real_mask, target_k]
y_E_real = df.loc[real_mask, target_E]

# Train/Test Split
X_train_sel, _, y_k_train, _ = train_test_split(X_real, y_k_real, test_size=0.2, random_state=42)
y_E_train = y_E_real.loc[X_train_sel.index]

def run_rfe(X, y, name, n=30):
    print(f">> Running RFE model for [{name}] ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    selector = RFE(estimator=rf, n_features_to_select=n, step=1)
    selector.fit(X_scaled, y)
    return X.columns[selector.support_].tolist()

# Perform separate filtering for the two targets.
sel_k = run_rfe(X_train_sel, y_k_train, "epsilon(k)", n=35)
sel_E = run_rfe(X_train_sel, y_E_train, "Young's Modulus", n=25)

# Take the union and forcibly retain the core physical terms.
final_set = set(sel_k + sel_E)
must_keeps = [col_porosity, col_porosity_acc, 'epsilon_ratio', 'APF', 'Covalent bond ratio (estimated)']
final_set.update([c for c in must_keeps if c in X_expanded.columns])
final_features = list(final_set)

# ==========================================
# Step 4. Save and generate heatmap
# ==========================================
df_final = pd.concat([df[["material_id", "formula", target_k, target_E, "data_source"]], X_expanded[final_features]], axis=1)
df_final.to_csv("materials_selected_features.csv", index=False)

print("\ngenerating heatmap...")
plot_data = df_final.loc[df_final['data_source'] == 0, [target_k, target_E] + final_features]
plt.figure(figsize=(28, 24))
sns.heatmap(plot_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 6})
plt.title('Final Feature Correlation Heatmap (Including All New Descriptors)', fontsize=20)
plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')

print(f"Finish！Features: {len(final_features)}.Heatmap saved.")


