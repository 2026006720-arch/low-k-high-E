import pandas as pd
from pymatgen.core import Composition
import joblib
import os
# Step 1. Load the saved model
model_path = "dielectric_modulus_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found: {model_path}，Please run the training script first！")

artifacts = joblib.load(model_path)
model = artifacts['model']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_cols']

# Step 2. Read the material data to be predicted
filename = "materials_selected_features2.csv"
df = pd.read_csv(filename)
def is_not_element(formula):
    try:
        # Check whether the number of elements in the chemical formula is greater than 1.
        return len(Composition(formula).elements) > 1
    except:
        return False
df_new = df[df['formula'].apply(is_not_element)].reset_index(drop=True)
print(f"The remaining data after excluding the elements: {len(df)} 行")

# Step 3. Preprocessing and prediction
X_new = df_new[feature_cols]
X_new_scaled = scaler.transform(X_new)
preds = model.predict(X_new_scaled)

df_new['predicted_k'] = preds[:, 0]
df_new['predicted_E_GPa'] = preds[:, 1]

# Step 4.Calculate the balance score and screen the optimal materials.
df_new['Score'] = (3 - df_new['predicted_k']) * (df_new['predicted_E_GPa'] / 10)

# 2. [span_2](start_span)Whether the marking meets the preferred criteria (k < 3.0 & E > 10 GPa)[span_2](end_span)
df_new['Is_Excellent'] = (df_new['predicted_k'] < 3.0) & (df_new['predicted_E_GPa'] > 10.0)

# 3. Sort by Score from high to low, with the best candidate material placed at the top.
df_new = df_new.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Step 5. Save results
output_file = "materials_predictions_results.csv"
df_new.to_csv(output_file, index=False)

print(f"Prediction and screening have been completed！The result has been saved to: {output_file}")
# [span_3](start_span)Print the top 20 candidate materials with the highest scores[span_3](end_span)
print("\nThe top 20 candidate materials with the highest scores：")
print(df_new[['formula','material_id', 'predicted_k', 'predicted_E_GPa', 'Score', 'Is_Excellent']].head(20))
