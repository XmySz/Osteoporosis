import glob
import os
from tqdm import tqdm
import natsort
import numpy as np
import pydicom
from PIL import Image


def read_and_save_dicom_image(input_filepath: str, output_filepath: str = None):
    """
    读取单张 DICOM 图像文件，并将其保存为带 .dcm 后缀的新文件。

    参数：
        input_filepath: 原始 DICOM 文件路径，可带或不带 .dcm 后缀。
        output_filepath: 保存路径；
            - 如果提供，确保以 .dcm 结尾；
            - 如果为 None，则自动在原始文件名后添加 .dcm 后缀。

    返回：
        image: numpy.ndarray, 经过 RescaleSlope/RescaleIntercept 校正的图像数据（float32）。
        ds: pydicom.dataset.FileDataset, 原始 DICOM 对象，包含所有元数据。
        saved_path: str, 最终保存的文件路径。
    """
    # 读取 DICOM
    try:
        ds = pydicom.dcmread(input_filepath, force=True)
    except Exception as e:
        raise IOError(f"无法读取 DICOM 文件: {input_filepath}\n原因: {e}")

    try:
        pixel_array = ds.pixel_array
    except AttributeError:
        raise AttributeError("无法在 DICOM 数据集中找到像素数据（Pixel Data）。")
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    slope = getattr(ds, 'RescaleSlope', 1.0)
    image = pixel_array.astype(np.float32) * slope + intercept

    if output_filepath:
        base, ext = os.path.splitext(output_filepath)
        if ext.lower() != '.dcm':
            output_filepath = base + '.dcm'
    else:
        base, _ = os.path.splitext(input_filepath)
        output_filepath = base + '.dcm'

    try:
        ds.save_as(output_filepath)
    except Exception as e:
        raise IOError(f"无法保存 DICOM 文件: {output_filepath}\n原因: {e}")

    return image, ds, output_filepath


def read_and_save_dicom_as_png(input_filepath: str, output_filepath: str = None, normalize: bool = True):
    """
    读取单张 DICOM 图像文件，并将像素数据保存为 PNG 格式图像。

    参数：
        input_filepath: 原始 DICOM 文件路径，可带或不带 .dcm 后缀。
        output_filepath: PNG 文件保存路径；
            - 如果提供，确保以 .png 结尾；
            - 如果为 None，则自动在原始文件名后添加 .png 后缀。
        normalize: 是否将像素数据线性归一化到 [0,255]（默认 True）；
            - 若 False，则直接截断到 [0,255] 并转换为 uint8。

    返回：
        image_array: numpy.ndarray, 保存前用于保存的二维 uint8 像素数据。
        ds: pydicom.dataset.FileDataset, 原始 DICOM 对象，包含所有元数据。
        saved_path: str, 最终保存的 PNG 文件路径。
    """
    try:
        ds = pydicom.dcmread(input_filepath, force=True)
    except Exception as e:
        raise IOError(f"无法读取 DICOM 文件: {input_filepath}\n原因: {e}")

    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except AttributeError:
        raise AttributeError("无法在 DICOM 数据集中找到像素数据（Pixel Data）。")

    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    slope = getattr(ds, 'RescaleSlope', 1.0)
    image = pixel_array * slope + intercept

    # 归一化或截断并转换为 uint8
    if normalize:
        img_min, img_max = np.min(image), np.max(image)
        if img_max > img_min:
            image_scaled = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image_scaled = np.zeros_like(image)
        image_uint8 = image_scaled.astype(np.uint8)
    else:
        image_clipped = np.clip(image, 0, 255)
        image_uint8 = image_clipped.astype(np.uint8)

    if output_filepath:
        base, ext = os.path.splitext(output_filepath)
        if ext.lower() != '.png':
            output_filepath = base + '.png'
    else:
        base, _ = os.path.splitext(input_filepath)
        output_filepath = base + '.png'

    try:
        img_pil = Image.fromarray(image_uint8)
        img_pil.save(output_filepath)
    except Exception as e:
        raise IOError(f"无法保存 PNG 文件: {output_filepath}\n原因: {e}")

    return image_uint8, ds, output_filepath


if __name__ == '__main__':
    filesdir = r"X:\医学影像智能计算及应用实验室资料\实验室数字病理资料\骨质疏松课题组资料\本院\21-24年骨松\骨松骨折\DX"
    files = natsort.natsorted(glob.glob(f"{filesdir}/*/*/*/I1000000"))
    for infile in tqdm(files, desc='进度'):
        # print(infile)
        img, ds_obj, out_path = read_and_save_dicom_as_png(infile, os.path.join(r"E:\Dataset\Jmszxyy\ZB", infile.split('\\')[-4] + '_' + infile.split('\\')[-2] + '.png'))
