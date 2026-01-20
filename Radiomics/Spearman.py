from pathlib import Path

import numpy as np
import pandas as pd


def remove_highly_correlated_features(data: pd.DataFrame, correlation_threshold: float = 0.9):
    """
    移除数据框中与其他特征高度相关的特征。

    计算所有数值特征之间的Spearman相关系数。
    对于绝对相关系数超过阈值的特征对，移除那个与所有其他特征
    具有更高平均绝对相关性的特征。

    Args:
        data (pd.DataFrame): 包含特征的数据框。'Patient' 和 'Target' 列会被忽略。
        correlation_threshold (float): 相关系数绝对值的阈值。超过此阈值的特征对将被处理。

    Returns:
        list: 经过筛选后要保留的特征列表。
    """
    # 1. 识别要进行相关性分析的特征列
    exclude_cols = ["Patient", "Target"]
    features = [col for col in data.columns if col not in exclude_cols]

    # 仅选择数值类型的特征进行相关性计算
    numeric_data = data[features].select_dtypes(include=np.number)
    numeric_features = numeric_data.columns.tolist()

    if len(numeric_features) < 2:
        print("警告: 数值特征少于2个，无法计算特征间相关性。将保留所有数值特征。")
        return numeric_features  # 返回找到的所有数值特征

    print(f"开始计算 {len(numeric_features)} 个数值特征间的Spearman相关性...")
    print(f"移除阈值: |Correlation| > {correlation_threshold}")

    # 2. 计算特征间的Spearman相关系数矩阵
    try:
        corr_matrix = numeric_data.corr(method='spearman')
    except Exception as e:
        print(f"计算相关性矩阵时出错: {e}")
        print("将保留所有数值特征。")
        return numeric_features

    # 获取相关性矩阵的绝对值
    abs_corr_matrix = corr_matrix.abs()

    # 3. 计算每个特征与其他所有特征的平均绝对相关性
    # 为了计算平均值，我们将对角线的1替换为NaN，这样它们在计算均值时会被忽略
    mean_abs_corr = abs_corr_matrix.copy()
    np.fill_diagonal(mean_abs_corr.values, np.nan)  # 使用 numpy 修改值
    mean_abs_corr = mean_abs_corr.mean(axis=0, skipna=True)  # 计算每列（每个特征）的平均值

    # 4. 找出要移除的特征
    features_to_remove = set()  # 使用集合避免重复添加

    # 遍历相关性矩阵的上三角（不包括对角线）
    for i in range(len(abs_corr_matrix.columns)):
        for j in range(i + 1, len(abs_corr_matrix.columns)):
            feature_i = abs_corr_matrix.columns[i]
            feature_j = abs_corr_matrix.columns[j]

            # 如果 feature_i 或 feature_j 已经在移除列表中，则跳过这对
            if feature_i in features_to_remove or feature_j in features_to_remove:
                continue

            # 检查相关性是否超过阈值
            if abs_corr_matrix.iloc[i, j] > correlation_threshold:
                print(f"发现高度相关对: '{feature_i}' 和 '{feature_j}' (Corr: {corr_matrix.iloc[i, j]:.4f})")

                # 比较这两个特征的平均绝对相关性
                mean_corr_i = mean_abs_corr[feature_i]
                mean_corr_j = mean_abs_corr[feature_j]

                # 移除平均绝对相关性较高的那个特征
                if mean_corr_i >= mean_corr_j:  # 如果相等，也移除i（或j，这里选择i）
                    features_to_remove.add(feature_i)
                    print(f"  -> 移除 '{feature_i}' (平均绝对相关性: {mean_corr_i:.4f} >= {mean_corr_j:.4f})")
                else:
                    features_to_remove.add(feature_j)
                    print(f"  -> 移除 '{feature_j}' (平均绝对相关性: {mean_corr_j:.4f} > {mean_corr_i:.4f})")

    # 5. 确定要保留的特征
    features_to_keep = [f for f in numeric_features if f not in features_to_remove]

    print(f"\n原始数值特征数量: {len(numeric_features)}")
    print(f"移除的高度相关特征数量: {len(features_to_remove)}")
    print(f"最终保留的特征数量: {len(features_to_keep)}")
    # print("移除的特征:", sorted(list(features_to_remove))) # 可以取消注释查看移除列表

    return features_to_keep


def main(file_path: str, corr_thresh: float = 0.9):
    """主执行函数"""
    try:
        try:
            data = pd.read_excel(file_path, sheet_name="Sheet1")
        except ValueError:
            print(f"警告: 在 {file_path} 中未找到名为 'Sheet1' 的工作表，尝试读取第一个工作表。")
            data = pd.read_excel(file_path, sheet_name=0)

        print(f"成功读取数据: {len(data)} 行, {len(data.columns)} 列")

        # 调用特征移除函数
        kept_features = remove_highly_correlated_features(data, correlation_threshold=corr_thresh)

        if not kept_features:
            print("没有特征被保留。")
            return

        output_cols = []
        if "Patient" in data.columns:
            output_cols.append("Patient")
        else:
            print("警告: 未在原始数据中找到 'Patient' 列。")

        # 检查并添加 Target 列（如果存在）
        if "Target" in data.columns:
            output_cols.append("Target")
        else:
            print("警告: 未在原始数据中找到 'Target' 列。")

        # 添加保留的特征
        output_cols.extend(kept_features)

        result_df = data[output_cols]

        output_stem = Path(file_path).stem
        if output_stem.endswith('_U检验筛选'):
            output_stem = output_stem[:-len('_U检验筛选')]
        if output_stem.endswith('_Spearman筛选'):
            output_stem = output_stem[:-len('_Spearman筛选')]
        # 添加新后缀
        output_suffix = f"_CorrRemoved_Spearman_thresh{corr_thresh}.xlsx"
        output_path = Path(file_path).parent / (output_stem + output_suffix)

        result_df.to_excel(output_path, index=False)
        print(f"\n包含筛选后特征的结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    excel_path_1 = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_O_U检验筛选.xlsx"
    excel_path_2 = r"D:\Data\PycharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Features_P_U检验筛选.xlsx"

    correlation_cutoff = 0.9

    print("-" * 50)
    print(f"方法: 移除高度相关特征 (Spearman)")
    print(f"阈值: |Correlation| > {correlation_cutoff}")
    print("-" * 50)

    main(excel_path_1, corr_thresh=correlation_cutoff)
    main(excel_path_2, corr_thresh=correlation_cutoff)
