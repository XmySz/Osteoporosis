import warnings
from pathlib import Path

import matplotlib.pyplot as plt  # 添加matplotlib导入
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer  # 处理可能的缺失值
from sklearn.linear_model import LassoCV  # 将LogisticRegressionCV替换为LassoCV
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
        output_path (str, optional): 图像保存路径。如果为None则只显示不保存。

    Returns:
        float: 交叉验证选出的最佳 alpha 值。
    """
    exclude_cols = ["Patient", "Target"]
    if "Target" not in data.columns: raise ValueError("数据框中未找到 'Target' 列")

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
    # 设置eps参数以生成更广范围的alpha值
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
    plt.errorbar(alphas, mean_mse, yerr=std_mse, fmt='o-', color='red', ecolor='lightgray',
                 elinewidth=3, capsize=0, markersize=5)

    # 添加最佳alpha的垂直线
    best_alpha = model_cv.alpha_
    plt.axvline(best_alpha, linestyle='--', color='black')

    # 设置对数刻度
    plt.xscale('log')
    plt.xlabel('Lambda (alpha)', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.title('LASSO Feature Selection', fontsize=16)

    # 添加网格
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # 紧凑布局
    plt.tight_layout()

    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_path}")

    plt.show()

    print(f'最佳alpha值:{best_alpha}')
    return best_alpha


def select_features_lasso_cv(data: pd.DataFrame, cv_folds: int = 5, n_alphas: int = 100,
                             random_state: int = 42, max_iter: int = 1000):
    """
    使用 Lasso (通过交叉验证选择alpha) 进行特征选择。

    Args:
        data (pd.DataFrame): 包含特征和 'Target' 列的数据框。
        cv_folds (int): 交叉验证的折数。
        n_alphas (int): 要尝试的 alpha 值的数量。LassoCV 会自动生成对数间隔的值。
                        alpha是正则化强度，较大 alpha 意味着更强的正则化 (更多系数被压缩为零)。
        random_state (int): 随机种子，用于可复现性。
        max_iter (int): 求解器的最大迭代次数。

    Returns:
        tuple: (selected_features, best_alpha, feature_importance_df)
            - selected_features (list): 通过 Lasso 选择的特征列表。
            - best_alpha (float): 交叉验证选出的最佳 alpha 值。
            - feature_importance_df (pd.DataFrame): 包含特征重要性信息的数据框。
    """
    exclude_cols = ["Patient", "Target"]
    if "Target" not in data.columns:
        raise ValueError("数据框中未找到 'Target' 列")
    if not pd.api.types.is_numeric_dtype(data["Target"]):
        raise ValueError("'Target' 列必须是数值类型")

    # 1. 分离特征和目标变量
    features = [col for col in data.columns if col not in exclude_cols]
    X = data[features]
    y = data["Target"]

    print(f"开始使用 Lasso 回归分析 {len(features)} 个特征...")
    print(f"使用 {cv_folds}-折交叉验证寻找最佳 alpha 值...")

    # 2. 识别并处理非数值特征 (这里选择仅保留数值特征)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_features) < len(features):
        non_numeric_features = [f for f in features if f not in numeric_features]
        print(f"警告: 检测到非数值特征，将跳过这些特征: {non_numeric_features}")
        X = X[numeric_features]
        features = numeric_features  # 更新特征列表
        if not features:
            print("错误: 没有可用的数值特征进行分析。")
            return [], np.nan, pd.DataFrame()

    # 3. 处理缺失值 (使用中位数填充)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=features)  # 转回DataFrame以保持列名

    # 4. 特征缩放 (至关重要!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    alphas = np.logspace(np.log10(1e-3), np.log10(10), n_alphas)

    # 5. 创建并拟合 LassoCV 模型
    # 通过交叉验证选择最佳 alpha
    model_cv = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=-1
    )

    try:
        model_cv.fit(X_scaled, y)
    except ValueError as e:
        print(f"拟合 LassoCV 时出错: {e}")
        print("可能的原因：数据中仍有问题（如全是NaN的列）。")
        return [], np.nan, pd.DataFrame()

    # 6. 获取最佳 alpha 值和对应的系数
    best_alpha = model_cv.alpha_
    coefficients = model_cv.coef_

    # 7. 选择系数不为零的特征
    # 由于浮点数精度问题，比较绝对值是否大于一个很小的阈值更安全
    threshold = 1e-5
    selected_features_mask = np.abs(coefficients) > threshold
    selected_features = X.columns[selected_features_mask].tolist()

    # 8. 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Is_Selected': selected_features_mask
    })

    # 计算标准化重要性 (将绝对系数值标准化到0-1之间)
    if len(selected_features) > 0:
        max_abs_coef = np.max(np.abs(coefficients[selected_features_mask]))
        feature_importance_df['Normalized_Importance'] = feature_importance_df['Abs_Coefficient'] / max_abs_coef
    else:
        feature_importance_df['Normalized_Importance'] = 0

    # 按绝对系数值降序排序
    feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

    print(f"\n交叉验证找到的最佳 alpha 值: {best_alpha:.6f}")
    print("\n特征重要性排序 (前10个):")
    print(feature_importance_df.head(10)[
              ['Feature', 'Coefficient', 'Abs_Coefficient', 'Normalized_Importance', 'Is_Selected']].round(4))

    print(f"\n筛选出 {len(selected_features)} 个显著特征 (基于 Lasso)")
    return selected_features, best_alpha, feature_importance_df


def main(file_path: str, cv: int = 5, n_alphas: int = 100, plot_path: bool = True):
    """主执行函数"""
    try:
        try:
            data = pd.read_excel(file_path, sheet_name="Sheet1")
        except ValueError:
            print(f"警告: 未找到名为 'Sheet1' 的工作表于 {file_path}，尝试读取第一个工作表。")
            data = pd.read_excel(file_path, sheet_name=0)

        print(f"成功读取数据: {len(data)} 行, {len(data.columns)} 列 from {file_path}")

        id_col = "Patient"
        if id_col not in data.columns:
            print(f"警告: 数据中未找到 '{id_col}' 列。输出将不包含此列。")
            id_col = None  # 标记为不存在
        if "Target" not in data.columns:
            print("错误: 数据中必须包含 'Target' 列。")
            return

        if plot_path:
            output_stem = Path(file_path).stem
            plot_output_path = Path(file_path).parent / f"{output_stem}_Lasso_Path.png"

            print("\n绘制Lasso正则化路径...")
            best_alpha = plot_lasso_path(data, n_alphas=n_alphas, cv_folds=cv, output_path=str(plot_output_path))
            print(f"最佳alpha值: {best_alpha:.6f}")

        selected, best_alpha_value, feature_importance_df = select_features_lasso_cv(data, cv_folds=cv,
                                                                                     n_alphas=n_alphas)

        if not selected:
            print("没有筛选出任何特征。")
            return

        output_stem = Path(file_path).stem

        # 保存筛选后的数据
        output_suffix = f"_LassoCV{cv}fold_筛选.xlsx"
        output_path = Path(file_path).parent / (output_stem + output_suffix)

        output_cols = []
        if id_col:  # 如果 Patient 列存在
            output_cols.append(id_col)
        output_cols.append("Target")
        output_cols.extend(selected)  # 添加筛选出的特征

        result_df = data[output_cols]

        # 创建包含重要性信息的特征表
        # 只包含被选中的特征
        selected_importance_df = feature_importance_df[feature_importance_df['Is_Selected']].copy()
        selected_importance_df = selected_importance_df[
            ['Feature', 'Coefficient', 'Abs_Coefficient', 'Normalized_Importance']].round(6)

        # 使用ExcelWriter保存多个工作表
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 工作表1: 筛选后的数据
            result_df.to_excel(writer, sheet_name='筛选数据', index=False)

            # 工作表2: 特征重要性信息（仅选中的特征）
            selected_importance_df.to_excel(writer, sheet_name='特征重要性', index=False)

            # 工作表3: 所有特征的重要性信息
            all_features_importance = feature_importance_df[
                ['Feature', 'Coefficient', 'Abs_Coefficient', 'Normalized_Importance', 'Is_Selected']].round(6)
            all_features_importance.to_excel(writer, sheet_name='所有特征重要性', index=False)

        print(f"\n结果已保存至: {output_path}")
        print("包含的工作表:")
        print("  - '筛选数据': 包含选中特征的原始数据")
        print("  - '特征重要性': 选中特征的重要性评分")
        print("  - '所有特征重要性': 所有特征的重要性评分和选择状态")

        # 另外保存一个单独的特征重要性文件
        importance_output_path = Path(file_path).parent / (
                output_stem + f"_特征重要性_alpha{best_alpha_value:.6f}.xlsx")
        all_features_importance.to_excel(importance_output_path, index=False)
        print(f"特征重要性详细信息已单独保存至: {importance_output_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
    except Exception as e:
        print(f"\n处理过程中发生意外错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的错误追踪信息


if __name__ == "__main__":
    excel_path_1 = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_O_CorrRemoved_Spearman_thresh0.9.xlsx"
    excel_path_2 = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_P_CorrRemoved_Spearman_thresh0.9.xlsx"

    cv_folds = 5
    num_alphas_to_try = 300

    print("-" * 50)
    print(f"方法: Lasso with {cv_folds}-Fold CV")
    print(f"尝试 alpha 值的数量: {num_alphas_to_try}")
    print("-" * 50)

    main(excel_path_1, cv=cv_folds, n_alphas=num_alphas_to_try)
    main(excel_path_2, cv=cv_folds, n_alphas=num_alphas_to_try)
