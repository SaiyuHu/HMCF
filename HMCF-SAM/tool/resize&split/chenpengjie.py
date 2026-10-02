import os
import pydicom
import SimpleITK as sitk
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from readmask import read_mask


def load_dicom_series_with_sitk_and_windowing(
        folder_path: str,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        normalize_to_255: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用 SimpleITK 读取 DICOM 序列，用 pydicom 获取窗宽窗位，返回 3D 体积和 slice 位置。

    Returns:
        volume: (H, W, D) uint8 array, [0, 255]
        slice_locations: (D,) array of z-positions in patient coordinate (mm)
        metadata: dict with spacing, origin, direction, windowing, etc.
    """
    # ==============================
    # Step 1: 用 SimpleITK 读取 3D 体积
    # ==============================
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)

    if not dicom_names:
        raise ValueError(f"No DICOM series found in {folder_path}")

    reader.SetFileNames(dicom_names)
    image_3d = reader.Execute()  # sitk.Image

    # 转为 numpy: SimpleITK 是 (W, H, D)，需转为 (H, W, D)
    volume_raw = sitk.GetArrayFromImage(image_3d)  # shape: (D, H, W)
    volume_raw = np.transpose(volume_raw, (1, 2, 0))  # -> (H, W, D)

    ds = pydicom.dcmread(dicom_names[0], force=True)

    # 获取 Rescale 参数
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))


    # ==============================
    # Step 2: 获取 slice locations（z 坐标）
    # ==============================
    # SimpleITK 提供了方向、原点、间距
    origin = np.array(image_3d.GetOrigin())  # (x, y, z)
    spacing = np.array(image_3d.GetSpacing())  # (sx, sy, sz)
    direction = np.array(image_3d.GetDirection()).reshape(3, 3)  # 方向矩阵

    D = volume_raw.shape[2]
    # 计算每个 slice 的 z 位置（在患者坐标系中）
    # z = origin[2] + i * spacing[2] * direction[2,2] （简化：假设轴对齐）
    # 更通用的方式：沿 slice normal 方向累加
    slice_normal = direction[:, 2]  # 第三列为 slice 法向量
    slice_spacing = spacing[2]

    # 位置 = origin + i * slice_spacing * slice_normal
    # 我们只关心沿法向量的投影（即解剖 z 位置）
    slice_locations = origin[2] + np.arange(D) * slice_spacing * slice_normal[2]
    slice_locations = slice_locations.astype(np.float32)

    # ==============================
    # Step 3: 用 pydicom 读取窗宽窗位
    # ==============================
    wc, ww = window_center, window_width
    if wc is None or ww is None:
        try:
            # 读取第一张 DICOM 获取窗宽窗位
            ds = pydicom.dcmread(dicom_names[0], force=True)
            wc = getattr(ds, 'WindowCenter', None)
            ww = getattr(ds, 'WindowWidth', None)

            # 处理多值情况
            if hasattr(wc, '__len__') and len(wc) > 0:
                wc = float(wc[0])
            elif wc is not None:
                wc = float(wc)

            if hasattr(ww, '__len__') and len(ww) > 0:
                ww = float(ww[0])
            elif ww is not None:
                ww = float(ww)

        except Exception as e:
            print(f"Warning: Failed to read windowing from {dicom_names[0]}: {e}")
            wc, ww = None, None

    # 如果仍无窗宽窗位，回退到数据范围
    if wc is None or ww is None:
        print("No windowing info found. Using raw intensity range.")
        vmin, vmax = volume_raw.min(), volume_raw.max()
        wc = (vmin + vmax) / 2
        ww = vmax - vmin

    # ==============================
    # Step 4: 应用窗宽窗位 + 归一化
    # ==============================
    img_min = wc - ww / 2
    img_max = wc + ww / 2
    # volume_raw = volume_raw * rescale_slope + rescale_intercept
    volume_windowed = np.clip(volume_raw, img_min, img_max)

    if normalize_to_255:
        vmin, vmax = img_min, img_max
        if vmax > vmin:
            volume_out = ((volume_windowed - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume_out = np.zeros_like(volume_windowed, dtype=np.uint8)
    else:
        volume_out = volume_windowed.astype(np.float32)

    # ==============================
    # Step 5: 构建 metadata
    # ==============================
    metadata = {
        'SeriesFileNames': list(dicom_names),
        'VolumeShape': volume_out.shape,  # (H, W, D)
        'Origin': origin,  # (x, y, z)
        'Spacing': spacing,  # (sx, sy, sz)
        'Direction': direction,  # 3x3 matrix
        'WindowCenterUsed': wc,
        'WindowWidthUsed': ww,
        'HasValidWindowing': (window_center is not None or window_width is not None),
        'PixelType': 'uint8' if normalize_to_255 else 'float32'
    }

    return volume_out, slice_locations, metadata


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    folder = "/Users/chenpengjie/Desktop/0002400024XZB/v"
    mask_path = "/Users/chenpengjie/Desktop/0002400024XZB/v/MASK-501"

    mask = read_mask(mask_path)

    mask_slice = np.argmax(np.sum(np.sum(mask, 0), 0))
    try:
        vol, locs, meta = load_dicom_series_with_sitk_and_windowing(
            folder,
            # window_center=40,   # 可手动覆盖
            # window_width=80,
            normalize_to_255=True
        )

        print(f"Volume shape: {vol.shape}")  # e.g., (512, 512, 120)
        print(f"Slice locations: {locs[:5]} ...")  # 物理位置 (mm)
        print(f"Spacing: {meta['Spacing']}")
        print(f"Window used: WC={meta['WindowCenterUsed']}, WW={meta['WindowWidthUsed']}")

    except Exception as e:
        print(f"Error: {e}")

    for i in np.where(np.sum(np.sum(mask, 0), 0)>0)[0]:
        plt.imshow(vol[ :, :, i], cmap='gray')
        plt.contour(mask[:, :, i])
        plt.show()

    print(0)

#vol是序列图像的矩阵，已经scale到0-255, locs是对应的每一层的slicelocation
#mask是0和1的二值矩阵
