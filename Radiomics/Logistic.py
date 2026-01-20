import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer  # 处理可能的缺失值
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler  # 特征缩放是必要的

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def select_features_logistic_l1_cv(data: pd.DataFrame, cv_folds: int = 5, cs_values=10, solver: str = 'liblinear',
                                   random_state: int = 42, max_iter: int = 1000):
    """
        使用 L1 正则化的 Logistic Regression (通过交叉验证选择C) 进行特征选择。

        Args:
            data (pd.DataFrame): 包含特征和 'Target' 列的数据框。
            cv_folds (int): 交叉验证的折数。
            cs_values (int or list): 要尝试的 C 值的数量或具体列表。C是正则化强度的倒数。
                                     较小 C 意味着更强的正则化 (更多系数被压缩为零)。
            solver (str): 用于优化的求解器，必须支持 L1 正则化 ('liblinear', 'saga')。
            random_state (int): 随机种子，用于可复现性。
            max_iter (int): 求解器的最大迭代次数。

        Returns:
            list: 通过 L1 Logistic Regression 选择的特征列表。
            float: 交叉验证选出的最佳 C 值。
    """
    exclude_cols = ['Patient', 'Target']

    features = [col for col in data.columns if col not in exclude_cols]
    X = data[features]
    y = data['Target']

    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 创建并训练模型
    model_cv = LogisticRegressionCV(
        Cs=cs_values,
        cv=cv_folds,
        penalty='l1',
        solver=solver,
        scoring='roc_auc',
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=-1
    )

    model_cv.fit(X_scaled, y)

    best_C = model_cv.C_[0]
    coefficients = model_cv.coef_[0]

    # 选择系数不为0的特征
    threshold = 1e-5
    selected_features_mask = np.abs(coefficients) > threshold
    selected_features = X.columns[selected_features_mask].to_list()

    print(f"\n交叉验证找到的最佳C值, {best_C:.4f}")
    print(pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})[selected_features_mask].round(4))

    return selected_features, best_C


def main(file_path: str, cv: int = 5, cs: int = 15, sol: str = 'liblinear'):
    try:
        data = pd.read_excel(file_path)

        required_col = ['Target']
        id_col = "Patient"

        selected, best_c_value = select_features_logistic_l1_cv(data, cv_folds=cv, cs_values=cs, solver=sol)

        if not selected:
            return

        output_stem = Path(file_path).stem
        output_suffix = '_LogRegL1筛选后.xlsx'
        output_path = Path(file_path).parent / (output_stem + output_suffix)

        output_cols = []
        if id_col:
            output_cols.append(id_col)
        output_cols.append('Target')
        output_cols.extend(selected)

        result_df = data[output_cols]

        result_df.to_excel(output_path, index=False)

    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
    except Exception as e:
        print(f"\n处理过程发生意外错误: {type(e.__name__): {str(e)}}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    excel_path = r"F:\Data\Jmzxyy\Osteoporosis\OriginalFeaturesAdded.xlsx"

    cv_folds = 5
    num_cs_to_try = 20
    solver_to_use = 'liblinear'

    main(excel_path, cv=cv_folds, cs=num_cs_to_try, sol=solver_to_use)
