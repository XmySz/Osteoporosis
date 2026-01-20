from pathlib import Path

import pandas as pd
import os, glob, shutil

excel_path = r'C:\Users\Administrator\Desktop\骨松-总表0516.xlsx'
# dirs_path = [
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\CT(2025.10.29)',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\CT导图（21年之前）\一人两次以上',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\CT导图（21年之前）\一人一次',
# ]

# dirs_path = [
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\14-16骨松\MR 腰椎',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\14-20年骨松（补）\MRI 腰椎',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\17-20年骨松\MRI 腰椎',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\21-24年骨松\骨松骨折\MRI',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\21-24年骨松\腰椎间盘突出+腰椎椎管狭窄\MRI',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\2014-2016脊柱骨科\腰椎间盘突出\MRI 腰椎',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\2014-2016脊柱骨科\腰椎椎管狭窄\MRI 腰椎 ',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\2017-2020脊柱骨科(2)\腰椎间盘突出\MRI 腰椎',
#     r'\\172.23.3.8\yxyxlab\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\2017-2020脊柱骨科(2)\腰椎椎管狭窄\MRI 腰椎',
# ]

dirs_path = [r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\新建文件夹']

names = pd.read_excel(excel_path)['住院号'].to_list()
names = [str(i) for i in names]
dirs = []

for dir in dirs_path:
    temp_dirs = glob.glob(dir+ '\\*')
    dirs.extend(temp_dirs)

index = 0
error_dirs = []
for dir in dirs:
    try:
        if dir.split('\\')[-1] in names:
            if not any(Path(os.path.join(r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\新建文件夹', dir.split('\\')[-1], 'CT')).iterdir()):
                print(dir)
                error_dirs.append(dir.split('\\')[-1])
                index += 1
    except:
        continue

pd.DataFrame.from_dict({'住院号': error_dirs}).to_excel(r'缺失住院号.xlsx', index=False)
