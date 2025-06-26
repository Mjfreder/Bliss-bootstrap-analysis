import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# ----------------------------
# Parameters
# ----------------------------
input_file = "Bliss_input.txt"
simple_output = "Simple_bliss.txt"
bootstrap_output = "bootstrap_bliss.txt"
heatmap_output = "Delta_Bliss_Median_Heatmap.png"
binmap_output = "Delta_Bliss_Binmap.png"
n_iterations = 1000

# ----------------------------
# Load data
# ----------------------------
df_raw = pd.read_csv(input_file, sep='\t')
drug_a_concs = df_raw.iloc[:, 0].values
drug_b_headers = df_raw.columns[1:]
drug_b_concs = [float(h.split('_')[0]) for h in drug_b_headers]
od_values = df_raw.iloc[:, 1:].values

drug_b_unique = sorted(set(drug_b_concs))
drug_a_unique = drug_a_concs

drug_b_index_map = {
    val: [i for i, b in enumerate(drug_b_concs) if b == val]
    for val in drug_b_unique
}
drug_a_index_map = {val: idx for idx, val in enumerate(drug_a_concs)}

# ----------------------------
# Compute OD means
# ----------------------------
od_mean_matrix = np.zeros((len(drug_a_unique), len(drug_b_unique)))
for i in range(len(drug_a_unique)):
    for j in range(len(drug_b_unique)):
        replicate_indices = drug_b_index_map[drug_b_unique[j]]
        od_mean_matrix[i, j] = np.mean([od_values[i, k] for k in replicate_indices])

# ----------------------------
# Control OD and inhibition
# ----------------------------
control_rows = np.where(drug_a_unique == 0)[0]
control_cols = [j for j, b in enumerate(drug_b_unique) if b == 0.0]
control_od_values = [od_mean_matrix[i, j] for i in control_rows for j in control_cols]
control_od = np.mean(control_od_values)

# Calculate inhibition and clip any negative values to 0
inhib_matrix = 100 * (1 - od_mean_matrix / control_od)
inhib_matrix = np.clip(inhib_matrix, 0, None)

# ----------------------------
# Simple ΔBliss matrix
# ----------------------------
delta_bliss_simple = np.full_like(inhib_matrix, 'X', dtype=object)
for i, a in enumerate(drug_a_unique):
    for j, b in enumerate(drug_b_unique):
        if a != 0 and b != 0:
            inhib_ab = inhib_matrix[i, j]
            inhib_a = inhib_matrix[i, drug_b_unique.index(0.0)]
            inhib_b = inhib_matrix[drug_a_unique.tolist().index(0.0), j]
            bliss_expected = inhib_a + inhib_b - (inhib_a * inhib_b / 100)
            delta_bliss_simple[i, j] = round(inhib_ab - bliss_expected, 4)

simple_bliss_df = pd.DataFrame(delta_bliss_simple, columns=[str(b) for b in drug_b_unique])
simple_bliss_df.insert(0, "Drugs", drug_a_concs)
simple_bliss_df.to_csv(simple_output, sep='\t', index=False)

# ----------------------------
# Bin scoring system
# ----------------------------
bin_to_score = {
    'Strong synergy': 3,
    'Moderate synergy': 2,
    'Weak synergy': 1,
    'Additive / No interaction': 0,
    'Weak antagonism': -1,
    'Moderate antagonism': -2,
    'Strong antagonism': -3
}
score_to_color = {
    3: '#8b0000',
    2: '#e66101',
    1: '#fdb863',
    0: '#cccccc',
    -1: '#92c5de',
    -2: '#4393c3',
    -3: '#2166ac'
}

# ----------------------------
# Bootstrapping + bin assignment (based on median)
# ----------------------------
bootstrap_results = []
bin_score_matrix = np.full((len(drug_a_unique), len(drug_b_unique)), 0)

for a, b in product(drug_a_unique, drug_b_unique):
    if a == 0 or b == 0:
        continue

    a_idx = drug_a_index_map[a]
    ab_idx = drug_b_index_map[b]
    a_only_idx = drug_b_index_map[0.0]
    b_only_idx = drug_b_index_map[b]

    ab_od_reps = [od_values[a_idx, i] for i in ab_idx]
    a_only_od_reps = [od_values[a_idx, i] for i in a_only_idx]
    b_only_od_reps = [od_values[drug_a_index_map[0.0], i] for i in b_only_idx]

    delta_bliss_samples = []
    for _ in range(n_iterations):
        od_ab = np.random.choice(ab_od_reps)
        od_a = np.random.choice(a_only_od_reps)
        od_b = np.random.choice(b_only_od_reps)

        inhib_ab = 100 * (1 - od_ab / control_od)
        inhib_a = 100 * (1 - od_a / control_od)
        inhib_b = 100 * (1 - od_b / control_od)
        bliss_expected = inhib_a + inhib_b - (inhib_a * inhib_b / 100)
        delta_bliss_samples.append(inhib_ab - bliss_expected)

    median_bliss = round(np.median(delta_bliss_samples), 4)
    q25 = round(np.percentile(delta_bliss_samples, 25), 4)
    q75 = round(np.percentile(delta_bliss_samples, 75), 4)

    # Assign bin based on median ΔBliss
    if median_bliss >= 10:
        assigned_bin = 'Strong synergy'
    elif median_bliss >= 5:
        assigned_bin = 'Moderate synergy'
    elif median_bliss >= 2:
        assigned_bin = 'Weak synergy'
    elif median_bliss > -2:
        assigned_bin = 'Additive / No interaction'
    elif median_bliss > -5:
        assigned_bin = 'Weak antagonism'
    elif median_bliss > -10:
        assigned_bin = 'Moderate antagonism'
    else:
        assigned_bin = 'Strong antagonism'

    score = bin_to_score[assigned_bin]
    bin_score_matrix[a_idx, drug_b_unique.index(b)] = score

    bootstrap_results.append({
        "Drug A": str(a),
        "Drug B": str(b),
        "Median": median_bliss,
        "Q25": q25,
        "Q75": q75,
        "Assigned Bin": assigned_bin
    })

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(bootstrap_output, sep='\t', index=False)

# ----------------------------
# Visual: Heatmap of Median ΔBliss (Correct Axis Order)
# ----------------------------
pivot_df = bootstrap_df.copy()
pivot_df["Drug A"] = pivot_df["Drug A"].astype(float)
pivot_df["Drug B"] = pivot_df["Drug B"].astype(float)

heatmap_data = pivot_df.pivot(index="Drug A", columns="Drug B", values="Median")
heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=True, fmt=".1f",
            cbar_kws={'label': 'Median ΔBliss'})
plt.title("Heatmap of Median ΔBliss (Bootstrapped, n=1000)")
plt.xlabel("Drug B Concentration")
plt.ylabel("Drug A Concentration")
plt.tight_layout()
plt.savefig(heatmap_output, dpi=300)
plt.close()

# ----------------------------
# Visual: Integer-coded Bin Map
# ----------------------------
cmap = sns.color_palette([score_to_color[i] for i in sorted(score_to_color.keys())])
plt.figure(figsize=(12, 8))
sns.heatmap(bin_score_matrix, annot=bin_score_matrix, fmt='d', cmap=cmap, center=0,
            xticklabels=drug_b_unique, yticklabels=drug_a_unique,
            cbar_kws={'label': 'Interaction Class'})
plt.title("Integer-coded ΔBliss Interaction Bin Map")
plt.xlabel("Drug B Concentration")
plt.ylabel("Drug A Concentration")
plt.tight_layout()
plt.savefig(binmap_output, dpi=300)
plt.close()
