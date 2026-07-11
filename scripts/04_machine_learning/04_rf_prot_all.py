# load libraries

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# load metadata
cedar = pd.read_csv("../final_predicted.csv")
cedar["epitope_id"] = cedar["epitope_id"].astype(str)

# drop notpresented/imm
before = len(cedar)
cedar = cedar[cedar["group"] != "notpresented_immunogenic"]
print("removed", before - len(cedar))
print(cedar["group"].value_counts())


# fast lookup table (replaces slow loop)
cedar_idx = cedar.set_index("epitope_id")

label_lookup = cedar_idx.to_dict("index")


# valid ids per dataset (avoid recomputing inside loop)
valid_ids_full = set(cedar["epitope_id"])


# load embeddings
data = np.load("../prottrans_final.npz", allow_pickle=True)

n = int(data["n_peptides"])
ids = np.array([str(x) for x in data["epitope_ids"]])

diff_eos_all = data["diff_eos"]

# map epitope_id -> embedding index (CRITICAL FIX)
id_to_idx = {eid: i for i, eid in enumerate(ids)}


# filters
def filter_full(df):
    return df

def filter_9mer(df):
    return df[df["mt_length"] == 9]

def filter_9mer_hla_a2(df):
    return df[(df["mt_length"] == 9) & (df["best_hla"] == "HLA-A*02:01")]


# main function
def run_rf_analysis(df, name):

    print("\nRUNNING:", name)

    valid_ids = set(df["epitope_id"])

    X_eos, X_mutpos, X_mean = [], [], []
    y, groups = [], []

    # build dataset
    for eid in valid_ids:

        if eid not in label_lookup:
            continue

        if eid not in id_to_idx:
            continue

        i = id_to_idx[eid]

        info = label_lookup[eid]

        mt_seq = str(info["mt_seq"])
        wt_seq = str(info["wt_seq"])

        mutpos = [p for p in range(len(mt_seq)) if mt_seq[p] != wt_seq[p]]

        if len(mutpos) != 1:
            continue

        mut_pos = mutpos[0]

        diff_perres = data[f"diff_perres_{i}"]

        if diff_perres.shape[0] != len(mt_seq):
            continue

        X_eos.append(diff_eos_all[i])
        X_mutpos.append(diff_perres[mut_pos])
        X_mean.append(diff_perres.mean(axis=0))

        y.append(int(info["label"]))
        groups.append(eid)

    y = np.array(y)
    groups = np.array(groups)

    X_eos = np.array(X_eos, dtype=np.float32)
    X_mutpos = np.array(X_mutpos, dtype=np.float32)
    X_mean = np.array(X_mean, dtype=np.float32)

    feature_sets = [
        ("diff_eos", X_eos),
        ("diff_mutpos", X_mutpos),
        ("diff_mean", X_mean),
    ]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12)

    splits = list(splitter.split(np.zeros(len(y)), y, groups))

    mean_fpr = np.linspace(0, 1, 100)

    results = []
    roc_curves = []

    # model loop
    for feat_name, X in feature_sets:

        print(feat_name)

        pr_scores = []
        roc_scores = []
        tprs = []

        for train_idx, test_idx in splits:

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=12,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            probs = rf.predict_proba(X_test)[:, 1]

            pr_scores.append(average_precision_score(y_test, probs))
            roc_scores.append(roc_auc_score(y_test, probs))

            fpr, tpr, _ = roc_curve(y_test, probs)

            if len(np.unique(fpr)) > 1:
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
            else:
                interp_tpr = np.zeros_like(mean_fpr)

            interp_tpr[0] = 0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[-1] = 1

        roc_curves.append({
            "dataset": name,
            "feature_set": feat_name,
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr
        })

        results.append({
            "plm": "ProtTrans",
            "dataset": name,
            "feature_set": feat_name,
            "pr_auc_mean": np.mean(pr_scores),
            "pr_auc_std": np.std(pr_scores),
            "roc_auc_mean": np.mean(roc_scores),
            "roc_auc_std": np.std(roc_scores),
            "pr_auc_chance": float(y.mean())
            })

    return results, roc_curves


# run all datasets
datasets = [
    ("full", filter_full),
    ("9mer", filter_9mer),
    ("9mer_HLA_A02", filter_9mer_hla_a2)
]

all_results = []
all_roc = []

for name, fn in datasets:

    subset_df = fn(cedar)

    results, roc = run_rf_analysis(subset_df, name)

    all_results.extend(results)
    all_roc.extend(roc)


# save results
pd.DataFrame(all_results).to_csv("prot_rf_all_results.csv", index=False)

with open("prot_rf_multi_rocs.pkl", "wb") as f:
    pickle.dump(all_roc, f)