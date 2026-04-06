import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Read data.
file1 = 'materials_clean.csv'
file2 = 'materials_features2.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df_merged = pd.merge(df1, df2, on='material_id', how='inner', suffixes=('', '_dup'))
df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_dup')]]

# 2. Save results
output_file = 'materials_with_features.csv'
df_merged.to_csv(output_file, index=False)
print(f"Merge completed, starting to prepare for imputation....")

# --- Perform physical constraint filtering here and keep NaNs for subsequent imputation. ---
df = pd.read_csv(output_file)
# Only remove values that are explicitly less than 0, while keeping NaN values  so that subsequent functions can perform predictive imputation.
df = df[~(
    (df['bulk_modulus_K_VRH (GPa)'] < 0) |
    (df['shear_modulus_G_VRH (GPa)'] < 0) |
    (df['youngs_modulus_E(GPa)'] < 0)
)].copy()
# --------------------------------------------------------

def scientific_imputation_optimized(df_input, n_trials=30):
    """
    Imputation for missing properties: K, G, and E.
    """
    if isinstance(df_input, str):
        df = pd.read_csv(df_input)
    else:
        # If a DataFrame object is passed in, use it directly.
        df = df_input.copy()

    # 1. Initialize flag (0 = original data, 1 = imputed data)
    if 'data_source' not in df.columns:
        df['data_source'] = 0

    # Target variable
    targets = ['bulk_modulus_K_VRH (GPa)', 'shear_modulus_G_VRH (GPa)', 'youngs_modulus_E(GPa)']

    # 2. Improve feature preparation logic.
    exclude_cols = targets + ['material_id', 'formula', 'data_source', 'structure', 'composition']

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_numeric = df[feature_cols].select_dtypes(include=[np.number])

    X_all = df_numeric.fillna(df_numeric.median())

    print(f"Feature preparation completed, totally {X_all.shape[1]} features。")

    for target in targets:
        if target not in df.columns:
            print(f"Skip {target}，column not found.")
            continue

        print(f"\n>>> Processing target: {target}")

        mask_real = df[target].notna()
        X_real = X_all[mask_real]
        y_real = df.loc[mask_real, target]
        X_missing = X_all[~mask_real]

        if len(X_missing) == 0:
            print(f"    This column has no missing values, skipping.")
            continue

        print(f"    Training sample: {len(X_real)} | To be predicted: {len(X_missing)}")

        # 3. Optuna parameter tuning
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
            model = XGBRegressor(**params)
            score = cross_val_score(model, X_real, y_real, cv=5, scoring='r2').mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_score = study.best_value
        print(f"    最佳 CV R2: {best_score:.4f}")

        # 4. Final imputation
        if best_score > 0.5:
            final_model = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
            final_model.fit(X_real, y_real)

            preds = final_model.predict(X_missing)
            preds = np.where(preds < 0.1, 0.1, preds)

            df.loc[~mask_real, target] = preds
            df.loc[~mask_real, 'data_source'] = 1
            print(f"    -> Imputation successful.")
        else:
            print(f"    -> Warning: R² is too low, skipping imputation for this column to ensure data quality.")

    return df


if __name__ == "__main__":
    df_result = scientific_imputation_optimized(df, n_trials=20)

    output_file = 'materials_features_filled.csv'
    df_result.to_csv(output_file, index=False)

    print(f"\n" + "=" * 40)
    print(f"All done！")
    print(f"Raw data: {len(df_result[df_result['data_source'] == 0])}")
    print(f"Imputed data: {len(df_result[df_result['data_source'] == 1])}")
    print(f"Save path: {output_file}")

