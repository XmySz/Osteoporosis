from typing import Union

import pandas as pd
import re
from collections import defaultdict


def analyze_excel_headers(input_filepath: str, output_filepath: str, sheet_name: Union[str, int] = 0) -> None:
    """
    读取一个Excel文件，统计其表头的分布情况。

    功能:
    1. 统计去除末尾 '_1', '_2', '_3', '_4' 后，每个基础表头出现的总次数。
    2. 记录每个基础表头分别出现在哪个后缀版本中（1, 2, 3, 4）。
    3. 将统计结果保存到一个新的Excel文件中。

    Args:
        input_filepath (str): 输入的XLSX文件路径。
        output_filepath (str): 输出统计结果的XLSX文件路径。
        sheet_name (str | int, optional): 要读取的工作表名称或索引。默认为第一个工作表 (0)。
    """
    try:
        df = pd.read_excel(input_filepath, sheet_name=sheet_name)
        headers = df.columns.tolist()
        print(f"成功读取文件 '{input_filepath}' 的工作表 '{sheet_name}'. 共 {len(headers)} 个表头。")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_filepath}' 未找到。")
        return
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return

    header_counts = defaultdict(int)
    header_versions = defaultdict(list)

    pattern = re.compile(r"^(.*?)_([1-4])$")

    # 3. 遍历所有表头进行分析
    for header in headers:
        match = pattern.match(str(header))
        if match:
            # 如果表头符合 '..._[1-4]' 的格式
            base_header = match.group(1)  # 提取基础表头
            version = int(match.group(2))  # 提取后缀数字

            header_counts[base_header] += 1
            if version not in header_versions[base_header]:
                header_versions[base_header].append(version)
        else:
            base_header = str(header)
            header_counts[base_header] += 1
            if 'N/A' not in header_versions[base_header]:
                header_versions[base_header].append('N/A')

    results = []
    for base_header in sorted(header_counts.keys()):
        count = header_counts[base_header]
        versions = sorted(header_versions[base_header])  # 对版本号列表也进行排序

        versions_str = ", ".join(map(str, versions))

        results.append({
            "基础表头": base_header,
            "出现总次数": count,
            "出现的版本号": versions_str
        })

    if not results:
        print("没有找到任何表头进行分析。")
        return

    output_df = pd.DataFrame(results)

    try:
        output_df.to_excel(output_filepath, index=False)
        print(f"统计结果已成功保存到 '{output_filepath}'。")
    except Exception as e:
        print(f"保存结果到Excel文件时发生错误: {e}")


if __name__ == "__main__":
    input_file = r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\20250618\total_LassoCV5fold_alpha0.008123_筛选.xlsx"
    output_file = "header_statistics.xlsx"

    analyze_excel_headers(input_file, output_file, sheet_name="筛选数据")
