import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def plot_lasso_path(data: pd.DataFrame, n_alphas: int = 100, cv_folds: int = 5,
                    random_state: int = 42, max_iter: int = 1000, output_path: str = None):
    """
    绘制Lasso的正则化路径，显示MSE随alpha变化的曲线。

    Args:
        data (pd.DataFrame): 包含特征和 'Target' 列的数据框。
        n_alphas (int): 要尝试的 alpha 值的数量。
        cv_folds (int): 交叉验证的折数。
        random_state (int): 随机种子。
        max_iter (int): 最大迭代次数。
        output_path (str, optional): 图像保存路径。如果为None则只显示不保存。a

    Returns:
        float: 交叉验证选出的最佳 alpha 值。
    """
    exclude_cols = ["Patient", "Target"]
    if "Target" not in data.columns:
        raise ValueError("数据框中未找到 'Target' 列")

    # 1. 分离特征和目标变量
    features = [col for col in data.columns if col not in exclude_cols]
    X = data[features]
    y = data["Target"]

    # 2. 识别并处理非数值特征
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_features) < len(features):
        non_numeric_features = [f for f in features if f not in numeric_features]
        print(f"警告: 检测到非数值特征，将跳过这些特征: {non_numeric_features}")
        X = X[numeric_features]

    # 3. 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # 4. 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 5. 创建并拟合LassoCV模型
    alphas = np.logspace(np.log10(1e-3), np.log10(10), n_alphas)
    model_cv = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=-1
    )

    model_cv.fit(X_scaled, y)

    # 6. 获取模型中的alphas和mse_path
    alphas = model_cv.alphas_
    mse_path = model_cv.mse_path_
    mean_mse = np.mean(mse_path, axis=1)
    std_mse = np.std(mse_path, axis=1)

    # 7. 绘制MSE随alpha变化的曲线
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, mean_mse, yerr=std_mse, fmt='o-', color='red', ecolor='lightgray', elinewidth=3, capsize=0,
                 markersize=5)

    # 添加最佳alpha的垂直线
    best_alpha = model_cv.alpha_
    plt.axvline(best_alpha, linestyle='--', color='black')

    # 设置对数刻度
    plt.xscale('log')
    plt.xlabel('Lambda (alpha)', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.title('LASSO Feature Selection', fontsize=16)

    # plt.xlabel('', fontsize=14)
    # plt.ylabel('', fontsize=14)
    # plt.title('', fontsize=16)

    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_path}")

    plt.show()

    print(f'最佳alpha值:{best_alpha}')
    return best_alpha


if __name__ == "__main__":
    data = pd.read_excel(r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\20250618\total.xlsx")
    plot_lasso_path(data, n_alphas=100, cv_folds=5,
                    output_path=r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\20250618\Lasso_Path.png")
