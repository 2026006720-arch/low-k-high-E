import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
import optuna
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Step 1. Read data
# ==========================================
filename = "materials_selected_features.csv"
df = pd.read_csv(filename)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print(f"Read data: {len(df)} rows. (source: {filename})")

target_k = 'epsilon(k)'
target_E = 'youngs_modulus_E(GPa)'
# Define feature columns
exclude_cols = ['material_id', 'formula', 'elements', 'data_source', target_k, target_E]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Number of features: {len(feature_cols)} ")

# ==========================================
# Step 2. Data split (6:2:2)
# ==========================================
df_real = df[df['data_source'] == 0].copy()
df_imputed = df[df['data_source'] != 0].copy()

n_real = len(df_real)
print(f" -> real data: {n_real}")
print(f" -> imputed data: {len(df_imputed)}")

real_temp, real_test = train_test_split(df_real, test_size=0.2, random_state=123)
real_train, real_val = train_test_split(real_temp, test_size=0.25, random_state=123)

train_set_init = pd.concat([real_train, df_imputed], axis=0).sample(frac=1, random_state=123)
val_set = real_val.copy()
test_set = real_test.copy()

X_train_res = train_set_init[feature_cols]
y_train_res = train_set_init[[target_k, target_E]]
X_val = val_set[feature_cols]
y_val = val_set[[target_k, target_E]]
X_test = test_set[feature_cols]
y_test = test_set[[target_k, target_E]]

print(f" -> [Train] Mixed training set: {len(X_train_res)} (include {len(real_train)}real data + {len(df_imputed)}imputed data)")
print(f" -> [Val]   Clean validation set: {len(X_val)} (only real)")
print(f" -> [Test]  Clean test set: {len(X_test)} (only real)")


# ==========================================
# Step 3. Cost-sensitive weights
# ==========================================
# Sample weights
def get_physics_weights(y):
    weights = np.ones(len(y))
    k_vals = y[target_k].values
    weights[k_vals < 3.0] = 5.0
    weights[k_vals < 2.5] = 12.0  # Enhance the contribution of low-K samples.
    return weights


train_weights = get_physics_weights(y_train_res)
print("\nCost-sensitive learning has been enabled.")

# ==========================================
# Step 4. Standardization
# ==========================================
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_res)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)


# ==========================================
# Step 5. Optuna automatic parameter tuning
# ==========================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
        "max_depth": trial.suggest_int("max_depth", 7, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 2.0),
        "n_jobs": -1,
        "random_state": 42
    }

    model = MultiOutputRegressor(XGBRegressor(**params, objective="reg:squarederror"))
    model.fit(X_train_scaled, y_train_res, sample_weight=train_weights)

    preds = model.predict(X_val_scaled)
    r2_k = r2_score(y_val[target_k], preds[:, 0])

    # If the model predicts k < 2 and E > 80 (unreasonably high modulus with low k), deduct points.
    phys_violation = np.mean((preds[:, 0] < 2.0) & (preds[:, 1] > 80))
    return r2_k - (phys_violation * 2.0)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print(f"Optuna parameter tuning，Optimal parameters: {best_params}")

# ==========================================
# Step 6. Final training
# ==========================================
final_model = MultiOutputRegressor(
    XGBRegressor(**best_params, objective="reg:squarederror", n_jobs=-1, random_state=42))
final_model.fit(X_train_scaled, y_train_res, sample_weight=train_weights)
model_artifacts = {
    'model': final_model,
    'scaler': scaler_X,
    'feature_cols': feature_cols
}
joblib.dump(model_artifacts, "dielectric_modulus_model.pkl")
print("\n[Success] The model, standardizer, and feature list have been saved to: dielectric_modulus_model.pkl")


# ==========================================
# Step 7. Evaluation
# ==========================================
print("\n===== Model Evaluation =====")


def quick_eval(X, y, name):
    preds = final_model.predict(X)
    y_k_true = y[target_k].values
    y_E_true = y[target_E].values
    y_k_pred = preds[:, 0]
    y_E_pred = preds[:, 1]

    r2_k = r2_score(y_k_true, y_k_pred)
    rmse_k = np.sqrt(mean_squared_error(y_k_true, y_k_pred))
    r2_E = r2_score(y_E_true, y_E_pred)
    rmse_E = np.sqrt(mean_squared_error(y_E_true, y_E_pred))



    print(f"[{name}]")
    print(f"  > Dielectric (k): R²={r2_k:.4f} ")
    print(f"  > Modulus (E)   : R²={r2_E:.4f} ")
    print("-" * 50)
    return preds


_ = quick_eval(X_train_scaled, y_train_res, "Train (Augmented)")
_ = quick_eval(X_val_scaled, y_val, "Validation")
preds_test = quick_eval(X_test_scaled, y_test, "Test (Final)")

# ==========================================
# Step 8. Plotting and SHAP analysis
# ==========================================
os.makedirs("figures", exist_ok=True)
pred_k = preds_test[:, 0]
pred_E = preds_test[:, 1]

# 1. Prediction performance comparison chart
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test[target_k], pred_k, alpha=0.6, edgecolors='w')
plt.plot([min(y_test[target_k]), 10], [min(y_test[target_k]), 10], 'r--', lw=2)
plt.xlabel('Measured k')
plt.ylabel('Predicted k')
plt.title(f'Dielectric Prediction\nR²={r2_score(y_test[target_k], pred_k):.3f}')

plt.subplot(1, 2, 2)
plt.scatter(y_test[target_E], pred_E, alpha=0.6, color='green', edgecolors='w')
plt.plot([0, 400], [0, 400], 'r--', lw=2)
plt.xlabel("Measured E (GPa)")
plt.ylabel("Predicted E (GPa)")
plt.title(f"Young's Modulus Prediction\nR²={r2_score(y_test[target_E], pred_E):.3f}")
plt.tight_layout(pad=4.0)
plt.savefig("figures/prediction_performance.png", dpi=300)

# 2. Pareto front analysis plot

# 3. SHAP analysis
X_train_df_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
explainer = shap.TreeExplainer(final_model.estimators_[0])
shap_values_full = explainer.shap_values(X_train_df_scaled)

# Global plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_full, X_train_df_scaled, show=False, max_display=15)
plt.tight_layout()
plt.savefig("figures/shap_summary_global.png", dpi=300)
plt.close()

# Group comparison chart
f_col = 'Group comparison chart'
f_related_cols = [c for c in feature_cols if 'fluorine' in c.lower() or 'F' in c]

if f_col in feature_cols:
    mask_has_F = X_train_res.reset_index(drop=True)[f_col] > 0

    fig = plt.figure(figsize=(20, 10))

    # --- left：with F ---
    ax1 = fig.add_subplot(1, 2, 1)
    shap.summary_plot(
        shap_values_full[mask_has_F.values],
        X_train_df_scaled[mask_has_F.values],
        show=False,
        plot_size=None
    )
    ax1.set_title("Mechanism: With Fluorine Content", fontsize=15, pad=20)

    # --- right：no F ---
    ax2 = fig.add_subplot(1, 2, 2)
    non_f_indices = ~mask_has_F.values
    X_no_f = X_train_df_scaled.iloc[non_f_indices].drop(columns=f_related_cols)
    f_col_indices = [feature_cols.index(c) for c in f_related_cols]
    shap_no_f = np.delete(shap_values_full[non_f_indices], f_col_indices, axis=1)

    shap.summary_plot(
        shap_no_f,
        X_no_f,
        show=False,
        plot_size=None
    )
    ax2.set_title("Mechanism: No Fluorine (Non-F Features Only)", fontsize=15, pad=20)

    plt.subplots_adjust(wspace=0.4)
    plt.savefig("figures/shap_group_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

