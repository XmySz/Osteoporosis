import logging
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostOptimizer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.best_params = None
        self.best_model = None
        self.best_score = 0

    def _objective(self, trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 42,
            'eta': trial.suggest_float('eta', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'gamma': trial.suggest_float('gamma', 0.01, 2.0),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'alpha': trial.suggest_float('alpha', 0.1, 5.0),
            'lambda': trial.suggest_float('lambda', 2.0, 5.0),
        }

        X_train, y_train = self.data_dict['train']
        X_val, y_val = self.data_dict['val']

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, dtrain,
            num_boost_round=5000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=500,
            callbacks=[optuna.integration.XGBoostPruningCallback(trial, 'eval-auc')],
            verbose_eval=False
        )

        # 如果当前模型比之前的最佳模型更好，就保存它
        if model.best_score > self.best_score:
            self.best_score = model.best_score
            self.best_model = model
            self.best_params = params.copy()
            logger.info(f"发现更好的模型 - AUC: {model.best_score:.4f}")

        return model.best_score

    def optimize(self, n_trials=500, save_path=None):
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            direction='maximize'
        )

        study.optimize(self._objective, n_trials=n_trials)

        logger.info(f"优化完成 - 最佳AUC: {self.best_score:.4f}")

        # 如果指定了保存路径，就保存模型到文件
        if save_path and self.best_model:
            self.best_model.save_model(save_path)
            logger.info(f"最佳模型已保存到: {save_path}")

        return self.best_model


def prepare_data(data_path):
    data = pd.read_excel(data_path)

    y = data["Target"]
    X = data.drop(["Target", "Patient", "SplitType"], axis=1, errors='ignore')

    train_mask = data["SplitType"] == "train"
    val_mask = data["SplitType"] == "valid"

    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    return {
        'train': (X_train_scaled, y_train),
        'val': (X_val_scaled, y_val)
    }


def main():
    data_path = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_O.xlsx"
    model_save_path = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\data\Radiomics_checkpoints\best_xgboost_model_O.json"

    data_dict = prepare_data(data_path)

    optimizer = XGBoostOptimizer(data_dict)
    best_model = optimizer.optimize(n_trials=200, save_path=model_save_path)

    return optimizer, best_model


if __name__ == "__main__":
    optimizer, model = main()