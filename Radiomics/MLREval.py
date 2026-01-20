import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, accuracy_score

model = load("Data/MM/best_logreg_model.joblib")
data = pd.read_excel(r"F:\Data\PycharmProjects\Jmzxyy\Osteoporosis\Radiomics\Data\total.xlsx", sheet_name='MM')

subset = 'all'

if subset == "all":
    mask = [True] * len(data)
else:
    mask = data["SplitType"] == subset

X = data[mask][['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4']]
y = data[mask]['Target']
probs = model.predict_proba(X)[:, 1]
preds = model.predict(X)

auc = roc_auc_score(y, probs)
acc = accuracy_score(y, preds)

print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")

result = pd.DataFrame({
    'FileName': data[mask]['Patient'],
    'Predicted_Probability': probs
})

result.to_csv(f"Data/MM/predictions_{subset}.csv", index=False)