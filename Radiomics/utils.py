import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


def normal_distribution_test(file_path: str):
    """
        正态分布检验函数
    """

    def select_features(data: pd.DataFrame, alpha: float = 0.05):
        """特征选择主函数"""
        exclude_cols = ["Patient", "Target"]
        features = [col for col in data.columns if col not in exclude_cols]
        significant_features = []

        print(f"开始分析 {len(features)} 个特征...")

        for feature in features:
            stat, p_value = shapiro(data[feature])

            print(f'统计量: {stat:.6f}, p值: {p_value:.6f}')
            if p_value > alpha:
                print(f'p值 > {alpha}，无法拒绝原假设，数据可能服从正态分布')
            else:
                print(f'p值 <= {alpha}，拒绝原假设，数据可能不服从正态分布')

        return significant_features

    try:
        data = pd.read_excel(file_path, sheet_name="Sheet1")
        print(f"成功读取数据: {len(data)} 行")

        selected = select_features(data)

        output_path = Path(file_path).parent / f"{Path(file_path).stem}_U检验筛选.xlsx"
        result_df = data[["Patient", "Target"] + selected]
        result_df.to_excel(output_path, index=False)
        print(f"结果已保存至: {output_path}")

    except Exception as e:
        print(f"错误: {str(e)}")


def dir_split_to_excel(file_path, save_path):
    results = {}
    train_files_0 = os.listdir(os.path.join(file_path, 'train', '0'))
    for train_file in train_files_0:
        results[train_file] = {'SplitType': 'train', 'Label': 0}

    train_files_1 = os.listdir(os.path.join(file_path, 'train', '1'))
    for train_file in train_files_1:
        results[train_file] = {'SplitType': 'train', 'Label': 1}

    valid_files_0 = os.listdir(os.path.join(file_path, 'test', '0'))
    for valid_file in valid_files_0:
        results[valid_file] = {'SplitType': 'valid', 'Label': 0}

    valid_files_1 = os.listdir(os.path.join(file_path, 'test', '1'))
    for valid_file in valid_files_1:
        results[valid_file] = {'SplitType': 'valid', 'Label': 1}

    df_data = []
    for filename, info in results.items():
        df_data.append([filename, info['SplitType'], info['Label']])

    pd.DataFrame(df_data, columns=['FileName', 'SplitType', 'Label']).to_excel(save_path, index=False)


def find_optimal_cutoff(excel_path: str, sheet_name: str, true_label_col: str, pred_proba_col: str) -> float:
    """
    根据约登指数从Excel文件中计算最佳分类阈值。
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    y_true = df[true_label_col]
    y_pred_proba = df[pred_proba_col]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)
    optimal_cutoff = thresholds[best_index]

    return optimal_cutoff


def calculate_metrics_auc_acc(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    n_bootstrap = 200
    auc_scores = []
    acc_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)
        auc_scores.append(roc_auc_score(y_true_boot, y_prob_boot))
        acc_scores.append(accuracy_score(y_true_boot, y_pred_boot))

    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    acc_ci = np.percentile(acc_scores, [2.5, 97.5])

    return f"{round(auc, 3)}[{round(auc_ci[0], 3)}-{round(auc_ci[1], 3)}]", f"{round(acc, 3)}[{round(acc_ci[0], 3)}-{round(acc_ci[1], 3)}]"


def calculate_metrics_sen_spe(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    n_bootstrap = 200
    sen_scores = []
    spe_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tn_boot = ((y_true_boot == 0) & (y_pred_boot == 0)).sum()
        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        sen_scores.append(tp_boot / (tp_boot + fn_boot))
        spe_scores.append(tn_boot / (tn_boot + fp_boot))

    sen_ci = np.percentile(sen_scores, [2.5, 97.5])
    spe_ci = np.percentile(spe_scores, [2.5, 97.5])

    return f"{round(sensitivity, 3)}[{round(sen_ci[0], 3)}-{round(sen_ci[1], 3)}]", f"{round(specificity, 3)}[{round(spe_ci[0], 3)}-{round(spe_ci[1], 3)}]"


def calculate_metrics_npv_ppv(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    n_bootstrap = 200
    ppv_scores = []
    npv_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tn_boot = ((y_true_boot == 0) & (y_pred_boot == 0)).sum()
        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        ppv_scores.append(tp_boot / (tp_boot + fp_boot))
        npv_scores.append(tn_boot / (tn_boot + fn_boot))

    ppv_ci = np.percentile(ppv_scores, [2.5, 97.5])
    npv_ci = np.percentile(npv_scores, [2.5, 97.5])

    return f"{round(ppv, 3)}[{round(ppv_ci[0], 3)}-{round(ppv_ci[1], 3)}]", f"{round(npv, 3)}[{round(npv_ci[0], 3)}-{round(npv_ci[1], 3)}]"


def calculate_metrics_f1(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    f1 = 2 * tp / (2 * tp + fp + fn)

    n_bootstrap = 200
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        f1_scores.append(2 * tp_boot / (2 * tp_boot + fp_boot + fn_boot))

    f1_ci = np.percentile(f1_scores, [2.5, 97.5])

    return f"{round(f1, 3)}[{round(f1_ci[0], 3)}-{round(f1_ci[1], 3)}]"


if __name__ == "__main__":
    # dir_split_to_excel(r'X:\qyj\Orthopedics\llm_FL\fl_data\0624data\已划分73_new\1234\CenterE(zd)', r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\split_ZD.xlsx")
    #

    auc_ci, acc_ci = calculate_metrics_auc_acc(
        r"F:\Data\Jmzxyy\职称申报骨质疏松中文课题\最终表格.xlsx",
        split_type='test1', cutoff=0.2786)
    print(auc_ci, acc_ci)

    sen_ci, spe_ci = calculate_metrics_sen_spe(
        r"F:\Data\Jmzxyy\职称申报骨质疏松中文课题\最终表格.xlsx",
        split_type="test1", cutoff=0.2786)
    print(sen_ci, spe_ci)

    ppv_ci, npv_ci = calculate_metrics_npv_ppv(
        r"F:\Data\Jmzxyy\职称申报骨质疏松中文课题\最终表格.xlsx",
        split_type="test1", cutoff=0.2786)
    print(ppv_ci, npv_ci)

    f1_ci = calculate_metrics_f1(
        r"F:\Data\Jmzxyy\职称申报骨质疏松中文课题\最终表格.xlsx",
        split_type="test1", cutoff=0.2786)
    print(f1_ci)
