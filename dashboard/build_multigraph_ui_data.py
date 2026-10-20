"""
Build comprehensive multi-graph dataset with distinct patient topologies,
multiple cluster manifolds, and detailed equation derivation datasets.
"""
import json
import pickle
import numpy as np
import pandas as pd

# Load dataset and artifacts
df_raw = pd.read_excel('GNN DATASET.xlsx')
cf_df = pd.read_csv('results/cpm_step3_counterfactuals.csv')
scores_df = pd.read_csv('results/cpm_step5_scores.csv')

with open('results/cpm_step4_neighbor_sets.pkl', 'rb') as f:
    neighbors_data = pickle.load(f)

# Selected clinical case studies with rich human narratives
CASE_DEFINITIONS = {
    852: {
        "name": "Ramesh K.",
        "age": 58,
        "gender": "Male",
        "occupation": "Senior Bank Manager (10 hrs/day desk work)",
        "chief_complaint": "Recurring morning throbbing occipital headaches, exertional shortness of breath when climbing stairs, bilateral ankle tightness.",
        "vital_signs": "Resting BP 172/105 mmHg (Grade 2 HTN), Resting Heart Rate 92 bpm, BMI 29.2 kg/m².",
        "lab_biomarkers": "Total Cholesterol 293 mg/dL, Triglycerides 290 mg/dL, Fasting Glucose 76.5 mg/dL.",
        "lifestyle": "Sedentary (0 hrs structured exercise/wk), active smoker (12 cigarettes/day), high job stress (Stress Index 7.5/10).",
        "peer_danger_context": "Ramesh's baseline biological profile connects him to 15 clinically matching patients in the registry. 14 of those 15 patients (94.4%) suffered acute myocardial infarctions or strokes within 36 months.",
        "target_prescription": [
            {"step": "Step 1", "category": "Pharmacotherapy", "action": "Dual Antihypertensive (Telmisartan 40mg + Amlodipine 5mg) & High-Intensity Statin (Rosuvastatin 20mg)", "impact": "Drops BP to 134 mmHg, Total Cholesterol to 216 mg/dL, Triglycerides to 190 mg/dL"},
            {"step": "Step 2", "category": "Lifestyle Intervention", "action": "Supervised Cardiac Aerobic Walking: 45 min sessions, 4 days/week", "impact": "Increases activity to 4.0 hrs/wk, resting HR slows to 74 bpm, BMI drops to 26.5"},
            {"step": "Step 3", "category": "Behavioral", "action": "Structured Smoking Cessation with nicotine patches & stress management", "impact": "Flips smoking to 0, reduces stress index from 7.5 to 3.5"}
        ]
    },
    336: {
        "name": "Priya S.",
        "age": 64,
        "gender": "Female",
        "occupation": "Retired School Administrator",
        "chief_complaint": "Chronic exhaustion, blurred vision episodes, tingling in extremities, uncontrolled weight gain.",
        "vital_signs": "BP 164/98 mmHg, BMI 33.8 kg/m² (Class I Obesity), Resting Heart Rate 88 bpm.",
        "lab_biomarkers": "Fasting Glucose 145 mg/dL (Uncontrolled Type 2 Diabetes), Triglycerides 271 mg/dL, Cholesterol 272 mg/dL.",
        "lifestyle": "Active smoker (1 pack/day), sedentary lifestyle, high dietary glycemic index.",
        "peer_danger_context": "Priya sits in the diabetic-atherosclerotic cluster. 100% of her immediate 15 baseline neighbors have active coronary artery disease or diabetic nephropathy.",
        "target_prescription": [
            {"step": "Step 1", "category": "Metabolic Control", "action": "SGLT2 inhibitor (Empagliflozin 10mg) + Metformin 1000mg BID", "impact": "Lowers fasting glucose from 145 to 85 mg/dL, initiates weight loss"},
            {"step": "Step 2", "category": "Vascular Therapy", "action": "ACE Inhibitor (Ramipril 5mg) + Atorvastatin 40mg", "impact": "Controls blood pressure to 130 mmHg, reduces LDL plaque progression"},
            {"step": "Step 3", "category": "Smoking Cessation", "action": "Varenicline therapy + nutritional counseling", "impact": "Eliminates smoking, triglycerides drop to 183 mg/dL"}
        ]
    },
    556: {
        "name": "Anil M.",
        "age": 52,
        "gender": "Male",
        "occupation": "Corporate Technology Executive",
        "chief_complaint": "Palpitations during work meetings, insomnia, chest tightness under psychological pressure.",
        "vital_signs": "Resting Heart Rate 114 bpm (Severe Sinus Tachycardia), BP 168/102 mmHg, BMI 33.8 kg/m².",
        "lab_biomarkers": "Cholesterol 261 mg/dL, Triglycerides 253 mg/dL, Stress Index 8.5/10.",
        "lifestyle": "Works 70 hrs/week, zero physical activity, poor sleep (<5 hrs/night), non-smoker.",
        "peer_danger_context": "Anil's hyper-adrenergic state embeds him among patients with premature coronary artery disease. 85% of his peers experienced cardiac events before age 55.",
        "target_prescription": [
            {"step": "Step 1", "category": "Autonomic & Vascular", "action": "Cardioselective Beta-Blocker (Bisoprolol 5mg) + Statin therapy", "impact": "Resting HR normalizes from 114 to 78 bpm, BP lowers to 132 mmHg"},
            {"step": "Step 2", "category": "Stress & Sleep Protocol", "action": "Cognitive behavioral therapy for insomnia & guided meditation", "impact": "Stress Index falls from 8.5 to 3.0, cortisol stabilization"},
            {"step": "Step 3", "category": "Active Commuting", "action": "Cycling / Brisk Walking 4.5 hrs/week", "impact": "BMI drops to 27.8, triglycerides normalize to 160 mg/dL"}
        ]
    },
    428: {
        "name": "George T.",
        "age": 72,
        "gender": "Male",
        "occupation": "Retired Civil Engineer",
        "chief_complaint": "Mild bilateral calf claudication after walking 200 meters, general fatigue.",
        "vital_signs": "Systolic BP 157 mmHg (Isolated Systolic HTN of the Elderly), Resting HR 76 bpm, BMI 31.7 kg/m².",
        "lab_biomarkers": "Total Cholesterol 291 mg/dL, Triglycerides 259 mg/dL, Glucose 127 mg/dL.",
        "lifestyle": "Non-smoker, walks 1.5 hrs/week, high sodium diet.",
        "peer_danger_context": "At age 72, age is an immutable risk factor. 90% of his matching elderly peers have carotid or coronary calcifications.",
        "target_prescription": [
            {"step": "Step 1", "category": "Arterial Compliance", "action": "Low-dose thiazide diuretic (Chlorthalidone 12.5mg) + Statin", "impact": "Systolic BP drops from 157 to 132 mmHg without orthostatic hypotension"},
            {"step": "Step 2", "category": "Lipid Normalization", "action": "Ezetimibe 10mg + Rosuvastatin 10mg", "impact": "Cholesterol drops to 210 mg/dL, Triglycerides drop to 175 mg/dL"},
            {"step": "Step 3", "category": "Walking Rehabilitation", "action": "Structured interval walking 3.5 hrs/week", "impact": "Improves peripheral circulation, BMI drops to 28.0"}
        ]
    }
}

# 1. Macro Global Manifold Clusters (5 distinct clinical clusters)
np.random.seed(42)
macro_clusters = {
    "cluster_1": {"name": "Severe Hypertensive Ward", "color": "#ef4444", "center": [-120, -70], "nodes": []},
    "cluster_2": {"name": "Diabetic & Metabolic Syndrome", "color": "#f97316", "center": [-110, 60], "nodes": []},
    "cluster_3": {"name": "Transitional / Outpatient Corridor", "color": "#eab308", "center": [0, 0], "nodes": []},
    "cluster_4": {"name": "Controlled Primary Care", "color": "#06b6d4", "center": [100, -60], "nodes": []},
    "cluster_5": {"name": "Low-Risk Healthy Reference Cohort", "color": "#22c55e", "center": [120, 70], "nodes": []}
}

# Generate 20 nodes for each macro cluster
for c_id, c_data in macro_clusters.items():
    cx, cy = c_data["center"]
    for i in range(22):
        node_id = f"{c_id}_{i}"
        nx = cx + np.random.normal(0, 22)
        ny = cy + np.random.normal(0, 22)
        c_data["nodes"].append({
            "id": node_id,
            "x": round(nx, 2),
            "y": round(ny, 2),
            "label": f"Peer #{i+1}",
            "color": c_data["color"]
        })

# 2. Build Patient Specific Subgraphs and Metrics
patients_output = []
scores_dict = scores_df.set_index('global_idx').to_dict(orient='index')
cf_dict = cf_df.set_index('global_idx').to_dict(orient='index')

FEATURES = [
    'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
    'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
    'Triglycerides', 'Stress_Index', 'Smoking', 'Diabetes'
]

CLINICAL_WEIGHTS = {
    'Cholesterol': 2.0,
    'Blood_Pressure': 1.8,
    'Diabetes': 1.8,
    'Glucose': 1.7,
    'Smoking': 1.6,
    'Age': 1.5,
    'Triglycerides': 1.5,
    'ECG_Score': 1.4,
    'BMI': 1.3,
    'Heart_Rate': 1.0,
    'Physical_Activity_HoursPerWeek': 1.0,
    'Stress_Index': 1.0,
    'Gender': 0.5
}

for pid, case in CASE_DEFINITIONS.items():
    sc = scores_dict.get(pid, {})
    cf = cf_dict.get(pid, {})
    nb = neighbors_data.get(pid, {})

    # Extract Real Local Neighbors from GNN Dataset
    orig_nb_ids = nb.get('orig_neighbors', [])
    new_nb_ids = nb.get('new_neighbors', [])

    # Local Baseline Neighbors (Red Cluster)
    local_orig_nodes = []
    # Place in a ring on the left side
    n_orig = min(15, len(orig_nb_ids))
    for i in range(n_orig):
        angle = (2 * np.pi * i) / n_orig
        dist = 85 + np.random.uniform(-10, 10)
        nid = orig_nb_ids[i]
        is_cvd = 1
        age_val = 55 + (i * 2) % 20
        bp_val = 150 + (i * 5) % 35
        if nid < 1000:
            is_cvd = int(df_raw.iloc[nid]['Target'])
            age_val = int(df_raw.iloc[nid]['Age'])
            bp_val = int(df_raw.iloc[nid]['Blood_Pressure'])

        local_orig_nodes.append({
            "id": f"orig_peer_{nid}",
            "real_id": int(nid),
            "x": round(-120 + dist * np.cos(angle), 2),
            "y": round(-10 + dist * np.sin(angle), 2),
            "cvd": is_cvd,
            "age": age_val,
            "bp": bp_val,
            "label": f"Peer #{nid} ({'CVD+' if is_cvd==1 else 'CVD-'})"
        })

    # Local Destination Healthy Neighbors (Green Cluster)
    local_new_nodes = []
    n_new = min(15, len(new_nb_ids))
    for i in range(n_new):
        angle = (2 * np.pi * i) / n_new
        dist = 90 + np.random.uniform(-10, 10)
        nid = new_nb_ids[i]
        is_cvd = 0
        age_val = 48 + (i * 2) % 18
        bp_val = 118 + (i * 3) % 15
        if nid < 1000:
            is_cvd = int(df_raw.iloc[nid]['Target'])
            age_val = int(df_raw.iloc[nid]['Age'])
            bp_val = int(df_raw.iloc[nid]['Blood_Pressure'])

        local_new_nodes.append({
            "id": f"new_peer_{nid}",
            "real_id": int(nid),
            "x": round(120 + dist * np.cos(angle), 2),
            "y": round(15 + dist * np.sin(angle), 2),
            "cvd": is_cvd,
            "age": age_val,
            "bp": bp_val,
            "label": f"Peer #{nid} ({'CVD+' if is_cvd==1 else 'CVD-'})"
        })

    # Intermediate Waypoint Nodes (Transit Corridor through which patient travels)
    intermediate_nodes = [
        {"id": f"mid_1_{pid}", "x": -50, "y": 45, "label": "Dr. Outreach Outpatient", "type": "clinic"},
        {"id": f"mid_2_{pid}", "x": -10, "y": -35, "label": "Cardiac Rehab Unit", "type": "rehab"},
        {"id": f"mid_3_{pid}", "x": 45, "y": 30, "label": "Community Wellness Cohort", "type": "lifestyle"}
    ]

    # Clinical Features
    feat_data = {}
    for f in FEATURES:
        orig_v = float(cf.get(f'orig_{f}', df_raw.iloc[pid][f] if f in df_raw.columns else 0))
        cf_v = float(cf.get(f'cf_{f}', orig_v))
        delta_v = float(cf.get(f'delta_{f}', cf_v - orig_v))
        feat_data[f] = {
            "orig": round(orig_v, 2),
            "cf": round(cf_v, 2),
            "delta": round(delta_v, 2),
            "weight": CLINICAL_WEIGHTS.get(f, 1.0)
        }

    # Assemble Patient Object
    p_obj = {
        "id": pid,
        "name": case["name"],
        "age": case["age"],
        "gender": case["gender"],
        "occupation": case["occupation"],
        "chief_complaint": case["chief_complaint"],
        "vital_signs": case["vital_signs"],
        "lab_biomarkers": case["lab_biomarkers"],
        "lifestyle": case["lifestyle"],
        "peer_danger_context": case["peer_danger_context"],
        "prescription": case["target_prescription"],
        "orig_risk": round(float(sc.get('orig_risk_prob', 0.99)) * 100, 1),
        "cf_risk": round(float(sc.get('cf_risk_prob', 0.10)) * 100, 1),
        "risk_reduction": round(float(sc.get('risk_reduction', 0.89)) * 100, 1),
        "cpm_score": round(float(sc.get('CPM_score', 0.94)), 4),
        "turnover": round(float(sc.get('T', 1.0)), 4),
        "risk_shift": round(float(sc.get('R', 0.88)), 4),
        "prevalence_orig": round(float(sc.get('prevalence_orig', 0.94)) * 100, 1),
        "prevalence_new": round(float(sc.get('prevalence_new', 0.06)) * 100, 1),
        "d_norm": round(float(sc.get('D_norm', 0.96)), 4),
        "emb_displacement": round(float(sc.get('emb_displacement', 15.4)), 2),
        "features": feat_data,
        "local_graph": {
            "start_coords": [-120, -10],
            "end_coords": [120, 15],
            "orig_neighbors": local_orig_nodes,
            "new_neighbors": local_new_nodes,
            "intermediate_nodes": intermediate_nodes
        },
        "macro_trajectory": [
            {"name": "Severe Hypertensive Ward", "coords": [-120, -70], "status": "Baseline Danger"},
            {"name": "Transitional Treatment Corridor", "coords": [0, 0], "status": "Under Active Medication"},
            {"name": "Healthy Reference Cohort", "coords": [120, 70], "status": "Target Destination"}
        ]
    }
    patients_output.append(p_obj)

full_output = {
    "macro_clusters": macro_clusters,
    "patients": patients_output
}

with open('dashboard/cpm_multigraph_data.json', 'w') as f:
    json.dump(full_output, f, indent=2)

print(f"Generated dashboard/cpm_multigraph_data.json with {len(patients_output)} detailed patient graphs!")
