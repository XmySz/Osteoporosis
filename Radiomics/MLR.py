import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === 1. 数据读取 ===
FILE_PATH = r"F:\Data\PycharmProjects\Jmzxyy\Osteoporosis\Radiomics\Data\total.xlsx"
data = pd.read_excel(FILE_PATH, sheet_name='MM')

y = data["Target"]
X = data.drop(["Target", "Patient", "SplitType", "Final", "New_Name"], axis=1, errors='ignore')

train_mask = data["SplitType"] == "Train"
val_mask = data["SplitType"] == "Valid"

X_train = X[train_mask]
X_valid = X[val_mask]
y_train = y[train_mask]
y_valid = y[val_mask]

# === 2. 定义 Optuna 目标函数 ===
def objective(trial):
    c = trial.suggest_loguniform("C", 1e-4, 1e3)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.TrialPruned()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    penalty=penalty,
                    solver=solver,
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=3407,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred)

    return auc_score


# === 3. 启动调参 ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print("最佳 AUC:", study.best_value)
print("最佳参数:", study.best_params)

# === 4. 用最佳参数重新训练最终模型 ===
best_params = study.best_params
final_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                random_state=42,
                **best_params,
            ),
        ),
    ]
)
final_model.fit(X_train, y_train)

valid_auc = roc_auc_score(y_valid, final_model.predict_proba(X_valid)[:, 1])
print(f"验证集 AUC: {valid_auc:.4f}")

# === 5. 保存模型 ===
dump(final_model, "Data/MM/best_logreg_model.joblib")
print("模型已保存")

# === 6. 使用 SHAP 计算特征贡献 ===
# 获取标准化后的数据
X_train_scaled = final_model.named_steps['scaler'].transform(X_train)
X_valid_scaled = final_model.named_steps['scaler'].transform(X_valid)

# 使用 LinearExplainer 专门处理线性模型
explainer = shap.LinearExplainer(final_model.named_steps['clf'], X_train_scaled)
shap_values = explainer.shap_values(X_valid_scaled)

# 计算每个特征的平均绝对 SHAP 值
shap_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Mean Absolute SHAP Value': np.abs(shap_values).mean(axis=0)
})

# 确保数据格式正确
shap_importance = shap_importance.sort_values(by='Mean Absolute SHAP Value', ascending=False)

# 绘制条形图
plt.figure(figsize=(12, 8))
plt.barh(shap_importance['Feature'][:15], shap_importance['Mean Absolute SHAP Value'][:15], color='skyblue')  # 只显示前15个特征
plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importance based on SHAP values')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n特征重要性排序:")
print(shap_importance.head(10))