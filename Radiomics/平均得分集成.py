import pandas as pd
from sklearn.metrics import roc_auc_score

file_path = r"F:\Data\PycharmProjects\Jmzxyy\Osteoporosis\Radiomics\Data\total.xlsx"

df = pd.read_excel(file_path)

valid_df = df[df['SplitType'] == 'valid'].copy()

# prediction_cols = ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4']
prediction_cols = ['Prediction3', 'Prediction4', 'Prediction2']
# prediction_cols = ['Prediction1']

valid_df['avg_prediction'] = valid_df[prediction_cols].mean(axis=1)

auc_score = roc_auc_score(valid_df['Target'], valid_df['avg_prediction'])

print(auc_score)
