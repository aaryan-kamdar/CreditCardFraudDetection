# 1 Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 2. Loading the dataset 

df=pd.read_csv("creditcard.csv")

# checking for skewed data before transformation 
print("\nSkew before transform:\n", df.skew(numeric_only=True))

# 3.1 Unskewing the amount column
df["Amount"] = np.log1p(df["Amount"])

# Time: convert raw seconds into hour-of-day (cyclical, more interpretable than
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop("Time", axis=1)
 
print("\nSkew after transform:\n", df.skew(numeric_only=True))


# 4. Splitting the data into features and targets and then train and test
x = df.drop("Class", axis=1)
y = df["Class"]
 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)   

# 5.1 Baseline Logistic regression
baseline_pipeline=ImbPipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(class_weight="balanced",max_iter=1000))
])
baseline_pipeline.fit(x_train,y_train)

baseline_preds = baseline_pipeline.predict(x_test)
baseline_probs = baseline_pipeline.predict_proba(x_test)[:, 1]
 
print("\n===== Class-Weighted Logistic Regression (baseline) =====")
print(classification_report(y_test, baseline_preds, target_names=["Not Fraud", "Fraud"]))


# 5.2 SMOTE AND three stratergies
sampling_strategies = {
    "SMOTE 10%": 0.1,          # fraud oversampled to 10% of majority class size
    "SMOTE 50%": 0.5,          # fraud oversampled to 50% of majority class size
    "SMOTE Full Balance": "minority"   # fraud oversampled to match majority 1:1
}
smote_results={}
results = {
    "Class-Weighted LR": {
        "pipeline": baseline_pipeline,
        "preds": baseline_preds,
        "probs": baseline_probs,
        "precision": precision_score(y_test, baseline_preds),
        "recall": recall_score(y_test, baseline_preds),
        "f1": f1_score(y_test, baseline_preds),
        "roc_auc": roc_auc_score(y_test, baseline_probs),
        "pr_auc": average_precision_score(y_test, baseline_probs)
    }
}
 
for name,strategy in sampling_strategies.items():
    pipeline=ImbPipeline([
    ("scaler",StandardScaler()),
    ("smote", SMOTE(sampling_strategy=strategy, random_state=42)),
    ("model",LogisticRegression(class_weight="balanced",max_iter=1000))     
    ])
    pipeline.fit(x_train, y_train)   # SMOTE only ever touches training data
 
    preds = pipeline.predict(x_test)
    probs = pipeline.predict_proba(x_test)[:, 1]
 
    results[name] = {
        "pipeline": pipeline,
        "preds": preds,
        "probs": probs,
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs)
    }
 
    print(f"\n===== {name} =====")
    print(classification_report(y_test, preds, target_names=["Not Fraud", "Fraud"]))
 

 # 5.3 Isolation forest
contamination_rate = y_train.mean()
 
iso_forest = IsolationForest(
    n_estimators=200, contamination=contamination_rate, random_state=42, n_jobs=-1
)
iso_forest.fit(x_train)   # no labels — learns "normal" from x_train alone
 
iso_raw_preds = iso_forest.predict(x_test)              # -1 = anomaly, 1 = normal
iso_preds = np.where(iso_raw_preds == -1, 1, 0)           # convert to 0/1 to match Class
iso_scores = -iso_forest.score_samples(x_test)            # higher = more anomalous
 
results["Isolation Forest"] = {
    "pipeline": iso_forest,
    "preds": iso_preds,
    "probs": iso_scores,
    "precision": precision_score(y_test, iso_preds),
    "recall": recall_score(y_test, iso_preds),
    "f1": f1_score(y_test, iso_preds),
    "roc_auc": roc_auc_score(y_test, iso_scores),
    "pr_auc": average_precision_score(y_test, iso_scores)
}
 
print("\n===== Isolation Forest =====")
print(classification_report(y_test, iso_preds, target_names=["Not Fraud", "Fraud"]))
 
# 6 Comparing all the models

comparison_df = pd.DataFrame({
    name: {
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1": res["f1"],
        "ROC-AUC": res["roc_auc"],
        "PR-AUC": res["pr_auc"]
    }
    for name, res in results.items()
}).T
 
print("\n===== Full Model Comparison (sorted by PR-AUC) =====")
print(comparison_df.sort_values("PR-AUC", ascending=False).round(4))


# 7 cost sensitive threshold 

best_model_name = comparison_df["PR-AUC"].idxmax()
print(f"\nBest model by PR-AUC: {best_model_name}")
 
best_probs = results[best_model_name]["probs"]
 
cost_false_negative = 500   # estimated cost of a missed fraud case — adjust to your assumptions
cost_false_positive = 5     # estimated cost of a false alarm (analyst review time)
 
precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
 
best_cost = float("inf")
best_threshold = 0.5
 
for t in thresholds:
    preds_at_t = (best_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_at_t).ravel()
    total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = t
 
print(f"\nCost-optimal threshold: {best_threshold:.4f}")
print(f"Estimated total cost at this threshold: ${best_cost:,}")
 
final_preds = (best_probs >= best_threshold).astype(int)
print("\n===== Final Model at Cost-Optimal Threshold =====")
print(classification_report(y_test, final_preds, target_names=["Not Fraud", "Fraud"]))
 
tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()
print(f"False Negatives (missed fraud): {fn} × ${cost_false_negative} = ${fn * cost_false_negative:,}")
print(f"False Positives (false alarms): {fp} × ${cost_false_positive} = ${fp * cost_false_positive:,}")
 
# 8 Saving the model 

joblib.dump(results[best_model_name]["pipeline"], "fraud_model.pkl")
joblib.dump(best_threshold, "fraud_threshold.pkl")
 
print(f"\nSaved {best_model_name} as fraud_model.pkl with threshold {best_threshold:.4f}")
