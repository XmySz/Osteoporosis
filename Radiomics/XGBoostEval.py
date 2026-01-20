import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostPredictor:
    def __init__(self, model_path):
        """
        初始化预测器

        Args:
            model_path: 保存的XGBoost模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """加载保存的XGBoost模型"""
        try:
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            logger.info(f"成功加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def prepare_data(self, data_path, fit_scaler=True):
        """
        准备预测数据，进行和训练时相同的预处理

        Args:
            data_path: 数据文件路径
            fit_scaler: 是否重新fit scaler（如果是测试集，应该用训练集的scaler参数）

        Returns:
            X_scaled: 标准化后的特征数据
            y: 目标变量
            patient_ids: 患者ID（如果存在）
        """
        data = pd.read_excel(data_path)

        # 提取患者ID（如果存在）
        patient_ids = data.get("Patient", pd.Series(range(len(data)), name="Patient"))

        # 提取目标变量
        y = data["Target"]

        # 提取特征，删除非特征列
        X = data.drop(["Target", "Patient", "SplitType"], axis=1, errors='ignore')

        # 标准化处理
        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("使用当前数据fit了新的StandardScaler")
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("使用已有的StandardScaler进行标准化")

        return X_scaled, y, patient_ids

    def predict(self, X):
        """
        使用模型进行预测

        Args:
            X: 输入特征数据

        Returns:
            predictions: 预测概率
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")

        # 转换为DMatrix格式
        dmatrix = xgb.DMatrix(X)

        # 进行预测
        predictions = self.model.predict(dmatrix)

        return predictions

    def evaluate(self, X, y_true, threshold=0.5):
        """
        评估模型性能

        Args:
            X: 输入特征
            y_true: 真实标签
            threshold: 分类阈值

        Returns:
            metrics: 包含各种评估指标的字典
        """
        # 获取预测概率
        y_pred_proba = self.predict(X)

        # 转换为二分类预测
        y_pred = (y_pred_proba >= threshold).astype(int)

        # 计算AUC
        auc = roc_auc_score(y_true, y_pred_proba)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 计算其他指标
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性/召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率
        accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率

        metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'confusion_matrix': cm,
            'predictions_proba': y_pred_proba,
            'predictions_binary': y_pred
        }

        return metrics

    def predict_and_save(self, data_path, output_path, fit_scaler=True):
        """
        对数据进行预测并保存结果

        Args:
            data_path: 输入数据路径
            output_path: 输出结果路径
            fit_scaler: 是否重新fit scaler
        """
        # 准备数据
        X_scaled, y_true, patient_ids = self.prepare_data(data_path, fit_scaler)

        # 评估模型
        metrics = self.evaluate(X_scaled, y_true)

        # 打印评估结果
        logger.info("=" * 50)
        logger.info("模型评估结果:")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"敏感性: {metrics['sensitivity']:.4f}")
        logger.info(f"特异性: {metrics['specificity']:.4f}")
        logger.info(f"精确率: {metrics['precision']:.4f}")
        logger.info("混淆矩阵:")
        logger.info(f"{metrics['confusion_matrix']}")
        logger.info("=" * 50)

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'Patient': patient_ids,
            'True_Label': y_true,
            'Predicted_Probability': metrics['predictions_proba'],
            'Predicted_Label': metrics['predictions_binary']
        })

        # 保存结果
        results_df.to_excel(output_path, index=False)
        logger.info(f"预测结果已保存到: {output_path}")

        return metrics, results_df


def main():
    # 配置参数
    config = {
        'model_path': r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\data\Radiomics_checkpoints\best_xgboost_model_O.json",
        'data_path': r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_O.xlsx",
        'output_path': "D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\O_Predict.xlsx",
        'fit_scaler': True  # 是否重新fit scaler（测试集通常设为False，但这里没有保存训练时的scaler）
    }

    try:
        # 创建预测器
        predictor = XGBoostPredictor(config['model_path'])

        # 进行预测和评估
        metrics, results_df = predictor.predict_and_save(
            config['data_path'],
            config['output_path'],
            config['fit_scaler']
        )

        return predictor, metrics, results_df

    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        raise


if __name__ == "__main__":
    predictor, metrics, results = main()
