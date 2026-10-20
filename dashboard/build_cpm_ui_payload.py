"""
Build a unified, self-contained JSON payload for the interactive CPM animation UI.
"""
import json
import pickle
import numpy as np
import pandas as pd

# 1. Load data
df_raw = pd.read_excel('GNN DATASET.xlsx')
cf_df = pd.read_csv('results/cpm_step3_counterfactuals.csv')
scores_df = pd.read_csv('results/cpm_step5_scores.csv')

with open('results/cpm_step4_neighbor_sets.pkl', 'rb') as f:
    neighbors_data = pickle.load(f)

# Feature list
FEATURES = [
    'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
    'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
    'Triglycerides', 'Stress_Index', 'Smoking', 'Diabetes',
    'Age', 'Gender', 'Family_History'
]

CLINICAL_WEIGHTS = {
    'Age': 1.5, 'Blood_Pressure': 1.8, 'Cholesterol': 2.0, 'Heart_Rate': 1.0,
    'BMI': 1.3, 'Diabetes': 1.8, 'Smoking': 1.6, 'Physical_Activity_HoursPerWeek': 1.0,
    'ECG_Score': 1.4, 'Family_History': 1.5, 'Glucose': 1.7, 'Triglycerides': 1.5,
    'Stress_Index': 1.0, 'Gender': 0.5
}

# 2. Extract 32D embeddings from neighbors_data to project with PCA/SVD
emb_list = []
idx_keys = list(neighbors_data.keys())

for k in idx_keys:
    emb_list.append(neighbors_data[k]['orig_emb'])
    emb_list.append(neighbors_data[k]['rebuilt_emb'])

emb_mat = np.array(emb_list, dtype=float)
# Center
mean_emb = emb_mat.mean(axis=0)
centered = emb_mat - mean_emb
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
coords_2d = np.dot(centered, Vt[:2].T)

# Scale coordinates to [-150, 150] space for canvas
c_min, c_max = coords_2d.min(axis=0), coords_2d.max(axis=0)
scale = 300.0 / (c_max - c_min + 1e-6)
coords_scaled = (coords_2d - (c_min + c_max)/2.0) * scale.min()

coord_map = {}
for i, k in enumerate(idx_keys):
    orig_c = coords_scaled[2*i].tolist()
    rebuilt_c = coords_scaled[2*i + 1].tolist()
    coord_map[k] = {
        'orig_x': round(orig_c[0], 2),
        'orig_y': round(orig_c[1], 2),
        'new_x': round(rebuilt_c[0], 2),
        'new_y': round(rebuilt_c[1], 2)
    }

# 3. Create synthetic background cluster nodes around the two poles (High Risk vs Low Risk)
np.random.seed(42)
background_nodes = []

# High-Risk Cluster center around mean orig_x, orig_y
orig_xs = [coord_map[k]['orig_x'] for k in idx_keys]
orig_ys = [coord_map[k]['orig_y'] for k in idx_keys]
center_high = [np.mean(orig_xs), np.mean(orig_ys)]

# Low-Risk Cluster center around mean new_x, new_y
new_xs = [coord_map[k]['new_x'] for k in idx_keys]
new_ys = [coord_map[k]['new_y'] for k in idx_keys]
center_low = [np.mean(new_xs), np.mean(new_ys)]

# Generate 30 high risk nodes
for i in range(30):
    background_nodes.append({
        'id': f'bg_h_{i}',
        'x': round(center_high[0] + np.random.normal(0, 28), 2),
        'y': round(center_high[1] + np.random.normal(0, 28), 2),
        'cvd': 1,
        'label': f'CVD+ Peer #{100+i}'
    })

# Generate 35 low risk nodes
for i in range(35):
    background_nodes.append({
        'id': f'bg_l_{i}',
        'x': round(center_low[0] + np.random.normal(0, 32), 2),
        'y': round(center_low[1] + np.random.normal(0, 32), 2),
        'cvd': 0,
        'label': f'Healthy Peer #{200+i}'
    })

# 4. Assemble Patients
patients = []
scores_by_idx = scores_df.set_index('global_idx').to_dict(orient='index')
cf_by_idx = cf_df.set_index('global_idx').to_dict(orient='index')

# Select top interesting clinical archetypes
archetype_names = {
    852: "Severe Hypertensive with Critical Hyperlipidemia",
    336: "Metabolic Syndrome & Heavy Smoker",
    428: "Sedentary Senior with High Triglycerides",
    556: "Borderline Diabetic with Vascular Strain",
    542: "Stage 2 Hypertension with Chronic Stress",
    953: "High-Risk Smoker with Severe BMI Elevation"
}

for p_idx in idx_keys:
    if p_idx not in scores_by_idx or p_idx not in cf_by_idx:
        continue
    
    sc = scores_by_idx[p_idx]
    cf = cf_by_idx[p_idx]
    nb = neighbors_data[p_idx]
    coords = coord_map[p_idx]
    
    # Feature delta summary
    feat_data = {}
    for f in FEATURES:
        orig_val = float(cf.get(f'orig_{f}', df_raw.iloc[p_idx][f] if f in df_raw.columns else 0))
        cf_val = float(cf.get(f'cf_{f}', orig_val))
        delta = float(cf.get(f'delta_{f}', cf_val - orig_val))
        feat_data[f] = {
            'orig': round(orig_val, 2),
            'cf': round(cf_val, 2),
            'delta': round(delta, 2),
            'weight': CLINICAL_WEIGHTS.get(f, 1.0)
        }

    # Clinical narrative
    archetype = archetype_names.get(p_idx, f"High-Risk Cardiac Patient #{p_idx}")
    
    p_data = {
        'id': int(p_idx),
        'name': archetype,
        'age': int(feat_data['Age']['orig']),
        'gender': 'Male' if feat_data['Gender']['orig'] == 1 else 'Female',
        'orig_risk': round(float(sc['orig_risk_prob']) * 100, 1),
        'cf_risk': round(float(sc['cf_risk_prob']) * 100, 1),
        'risk_reduction': round(float(sc['risk_reduction']) * 100, 1),
        'cpm_score': round(float(sc['CPM_score']), 4),
        'turnover': round(float(sc['T']), 4),
        'prevalence_orig': round(float(sc['prevalence_orig']) * 100, 1),
        'prevalence_new': round(float(sc['prevalence_new']) * 100, 1),
        'risk_shift': round(float(sc['R']), 4),
        'emb_displacement': round(float(sc['emb_displacement']), 2),
        'd_norm': round(float(sc['D_norm']), 4),
        'f_norm': round(float(sc['F_norm']), 4),
        'n_features_changed': int(sc['n_features_changed']),
        'coords': coords,
        'features': feat_data,
        'orig_neighbor_count': len(nb['orig_neighbors']),
        'new_neighbor_count': len(nb['new_neighbors']),
        'orig_cvd_count': int(round(len(nb['orig_neighbors']) * float(sc['prevalence_orig']))),
        'new_cvd_count': int(round(len(nb['new_neighbors']) * float(sc['prevalence_new'])))
    }
    patients.append(p_data)

# Sort patients by CPM score descending
patients.sort(key=lambda x: x['cpm_score'], reverse=True)

final_payload = {
    'weights': {
        'turnover': 0.40,
        'risk_shift': 0.40,
        'embedding': 0.20
    },
    'clinical_weights': CLINICAL_WEIGHTS,
    'background_nodes': background_nodes,
    'center_high': center_high,
    'center_low': center_low,
    'patients': patients
}

with open('dashboard/cpm_ui_payload.json', 'w') as f:
    json.dump(final_payload, f, indent=2)

print(f"Successfully generated dashboard/cpm_ui_payload.json with {len(patients)} patients!")
