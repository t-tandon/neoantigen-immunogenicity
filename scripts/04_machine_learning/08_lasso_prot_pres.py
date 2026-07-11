# load libraries

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# load metadata
cedar = pd.read_csv("final_predicted.csv")
cedar["epitope_id"] = cedar["epitope_id"].astype(str)

# drop notpresented immunogenic
before = len(cedar)
cedar = cedar[cedar["group"] != "notpresented_immunogenic"]

print("removed:", before - len(cedar))
print(cedar["group"].value_counts())


# fast lookup
cedar_idx = cedar.set_index("epitope_id")
label_lookup = cedar_idx.to_dict("index")


# load embeddings
data = np.load("prottrans_final.npz", allow_pickle=True)

n = int(data["n_peptides"])
ids = np.array([str(x) for x in data["epitope_ids"]])

diff_eos_all = data["diff_eos"]

#id → index mapping
id_to_idx = {eid: i for i, eid in enumerate(ids)}


# filters (3 datasets like your other scripts)
def filter_full(df):
    return df


datasets = [
    ("full", filter_full),
]


splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12)
mean_fpr = np.linspace(0, 1, 100)


all_results = []
all_roc = []


# run all datasets
for dataset_name, fn in datasets:
    print("DATASET:", dataset_name)

    df = fn(cedar)
    valid_ids = set(df["epitope_id"])

    X_eos, X_mutpos, X_mean = [], [], []
    X_pres = [] 
    y, groups = [], []

    lengths, hlas = [], []
    from_aas, to_aas, positions = [], [], []

    kept = 0

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

        if len(mt_seq) != len(wt_seq):
            continue

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
        X_pres.append([
            float(info["delta_affinity"]), 
            float(info["delta_presentation_score"])
        ])

        y.append(int(info["label"]))
        groups.append(eid)

        lengths.append(info["mt_length"])
        hlas.append(info["best_hla"])

        from_aas.append(wt_seq[mut_pos])
        to_aas.append(mt_seq[mut_pos])
        positions.append(mut_pos)

        kept += 1

    print("kept:", kept)

    y = np.array(y)
    groups = np.array(groups)

    X_eos = np.array(X_eos, dtype=np.float32)
    X_mutpos = np.array(X_mutpos, dtype=np.float32)
    X_mean = np.array(X_mean, dtype=np.float32)
    X_pres = np.array(X_pres, dtype=np.float32)

    print("baseline PR-AUC:", y.mean())


    #confound baseline
    conf_table = pd.DataFrame({"length": lengths, "hla": hlas})
    conf_dummy = pd.get_dummies(conf_table, columns=["hla"])
    X_confound = conf_dummy.to_numpy(dtype=np.float32)

    #substitution baseline
    sub_table = pd.DataFrame({
        "from_aa": from_aas,
        "to_aa": to_aas,
        "position": positions
    })

    sub_dummy = pd.get_dummies(sub_table, columns=["from_aa", "to_aa"])
    X_subst = sub_dummy.to_numpy(dtype=np.float32)

    #original views
    feature_sets = [
    ("presentation_only", X_pres),   #presentation only

    #eos
    ("diff_eos", X_eos),
    ("diff_eos_pres", np.hstack([X_eos, X_pres])),

    #mutpos 
    ("diff_mutpos", X_mutpos),
    ("diff_mutpos_pres", np.hstack([X_mutpos, X_pres])),

    #mean 
    ("diff_mean", X_mean),
    ("diff_mean_pres", np.hstack([X_mean, X_pres]))]


    splits = list(splitter.split(np.zeros(len(y)), y, groups))


    results = []
    roc_data = []


    for feat_name, X in feature_sets:

        print("\nfeature:", feat_name)

        pr_scores = []
        roc_scores = []
        tprs = []

        for train_idx, test_idx in splits:

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1, 10],
                    cv=3,
                    penalty="l1",
                    solver="saga",
                    scoring="average_precision",
                    class_weight="balanced",
                    max_iter=5000,
                    n_jobs=-1
                )
            )

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            pr_scores.append(average_precision_score(y_test, probs))
            roc_scores.append(roc_auc_score(y_test, probs))

            fpr, tpr, _ = roc_curve(y_test, probs)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)


        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[-1] = 1


        roc_data.append({
            "dataset": dataset_name,
            "feature_set": feat_name,
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr
        })


        results.append({
            "plm": "ProtTrans",
            "dataset": dataset_name,
            "feature_set": feat_name,
            "pr_auc_mean": np.mean(pr_scores),
            "pr_auc_std": np.std(pr_scores),
            "roc_auc_mean": np.mean(roc_scores),
            "roc_auc_std": np.std(roc_scores),
            "pr_auc_chance": float(y.mean())
        })


    all_results.extend(results)
    all_roc.extend(roc_data)


#save outputs
pd.DataFrame(all_results).to_csv("lasso_prot_pres.csv", index=False)

with open("roc_prot_pres.pkl", "wb") as f:
    pickle.dump(all_roc, f)