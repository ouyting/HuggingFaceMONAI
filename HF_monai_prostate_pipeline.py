import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

# MONAI 核心组件
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CenterSpatialCropd, ScaleIntensityd, NormalizeIntensityd, EnsureTyped, Invertd,
    KeepLargestConnectedComponent, FillHoles,
)
from monai.networks.nets import UNet
from monai.bundle import download

# 从 Hugging Face 的 MONAI 官方仓库下载前列腺解剖分割 Bundle 模型
# 该模型输出 背景(0) / 中央腺体CG(1) / 外周带PZ(2)，这里同时输出 CG、PZ 及二者合并的整个前列腺
MODEL_NAME = "prostate_mri_anatomy"
MODEL_DIR = "./models"
BUNDLE_ROOT = os.path.join(MODEL_DIR, MODEL_NAME)
# 训练数据在轴位平面每侧中心裁剪 20%（center_crop.py 的默认 margin=0.2）
CROP_MARGIN = 0.2
# 可视化时各结构的轮廓样式：(名称, 颜色, 线宽)
CONTOUR_STYLES = [("Prostate", "lime", 3.0), ("CG", "red", 1.2), ("PZ", "deepskyblue", 1.2)]


def load_model(device):
    """下载（如需要）并载入 Bundle 的预训练 UNet。"""
    if not os.path.exists(os.path.join(BUNDLE_ROOT, "models", "model.pt")):
        print("正在从 Hugging Face 下载 MONAI 官方前列腺分割 Bundle...")
        download(name=MODEL_NAME, bundle_dir=MODEL_DIR)

    # 按 Bundle 的 network_def 构建 UNet 并载入预训练权重
    print("正在载入前列腺 3D 分割网络权重...")
    net = UNet(
        spatial_dims=3, in_channels=1, out_channels=3,
        channels=(16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2),
        num_res_units=4, norm="batch", act="prelu", dropout=0.15,
    ).to(device)
    state = torch.load(os.path.join(BUNDLE_ROOT, "models", "model.pt"), map_location=device)
    net.load_state_dict(state.get("model", state))
    net.eval()
    return net


def segment(mri_path, net, device):
    """对 T2W 轴位前列腺 MRI（.nii.gz）做分割，返回 (orig_img, mask, cg, pz, zones)，均在原始图像空间。"""
    # 预处理流水线：必须与 Bundle 训练时一致（见 configs/inference.json 与 docs/README.md）
    orig_img = nib.load(mri_path)
    orig_shape = orig_img.shape[:3]
    preprocessing = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        CenterSpatialCropd(keys="image", roi_size=(
            int(orig_shape[0] * (1 - 2 * CROP_MARGIN)),
            int(orig_shape[1] * (1 - 2 * CROP_MARGIN)),
            -1,
        )),
        # 将空间分辨率统一重采样到 [0.5, 0.5, 0.5] mm
        Spacingd(keys="image", pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
        # MR 灰度没有物理单位，不能用 CT 的窗宽窗位截断，而是先缩放到 [0,1] 再做 Z-score 标准化
        ScaleIntensityd(keys="image", minv=0, maxv=1),
        NormalizeIntensityd(keys="image"),
        EnsureTyped(keys="image"),
    ])
    data = preprocessing({"image": mri_path})
    input_tensor = data["image"].unsqueeze(0).to(device)

    # 显卡加速推理
    print("显卡正在执行 3D 前列腺体积分割推理...")
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        # 滑动窗口重叠 50% + 高斯加权，消除窗口拼接处的断痕
        logits = monai.inferers.sliding_window_inference(
            inputs=input_tensor,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=net,
            overlap=0.5,
            mode="gaussian",
        )
        probs = torch.softmax(logits.float(), dim=1)

    # 将 CG/PZ 两通道概率反变换回原始图像空间后再二值化，边界更平滑
    # （不反变换背景通道：裁剪的逆变换会在裁剪区外补 0，背景概率为 0 会被误判为前列腺）
    data["pred"] = probs[0, 1:].cpu()
    data = Invertd(keys="pred", transform=preprocessing, orig_keys="image", nearest_interp=False)(data)
    probs_orig = data["pred"]  # [0]=CG, [1]=PZ
    # 整个前列腺的概率 = CG 概率 + PZ 概率（即 1 - 背景概率）
    mask = (probs_orig.sum(dim=0, keepdim=True) > 0.5).to(torch.uint8)
    mask = KeepLargestConnectedComponent()(mask)  # 去除假阳性碎块
    mask = FillHoles()(mask)                      # 填补腺体内部空洞
    mask = mask[0].numpy().astype(np.uint8)

    # 每层轴位切片只保留面积最大的轮廓并填洞，去除腺体底部精囊等与主体 3D 相连的小碎块
    axial_axis = int(np.where(nib.orientations.io_orientation(orig_img.affine)[:, 0] == 2)[0][0])
    mask = np.moveaxis(mask, axial_axis, -1)
    for z in range(mask.shape[-1]):
        labels = measure.label(mask[..., z], connectivity=2)
        if labels.max() > 1:
            largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
            mask[..., z] = (labels == largest).astype(np.uint8)
        mask[..., z] = ndimage.binary_fill_holes(mask[..., z]).astype(np.uint8)  # 填补该层内的空洞
    mask = np.moveaxis(mask, -1, axial_axis)

    # 在整个前列腺掩膜内按 CG / PZ 概率大小划分分区，保证 CG ∪ PZ 与整个前列腺完全一致
    probs_np = probs_orig.numpy()
    cg = ((probs_np[0] >= probs_np[1]) & (mask > 0)).astype(np.uint8)
    if cg.any():
        cg = KeepLargestConnectedComponent()(cg[None])[0].numpy().astype(np.uint8)  # CG 是单一实体，碎块归入 PZ
        cg = ndimage.binary_fill_holes(cg).astype(np.uint8)  # CG 内部的孤立 PZ 体素归回 CG
    pz = ((mask > 0) & (cg == 0)).astype(np.uint8)
    zones = (cg * 1 + pz * 2).astype(np.uint8)  # 0=背景, 1=CG, 2=PZ
    return orig_img, mask, cg, pz, zones


def voxel_volume_ml(orig_img):
    return np.prod(orig_img.header.get_zooms()[:3]) / 1000.0


def to_lps(orig_img, *arrays):
    """统一转到 LPS 方向，使 slice.T 显示为放射学视图（前方朝上、患者右侧在图像左侧）。"""
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(orig_img.affine), nib.orientations.axcodes2ornt(("L", "P", "S")))
    return [nib.orientations.apply_orientation(a, ornt) for a in arrays]


def draw_slice(ax, image_lps, masks_lps, z):
    """在一张轴位切片上绘制整个前列腺、CG、PZ 的边界轮廓。masks_lps 顺序与 CONTOUR_STYLES 一致。"""
    ax.imshow(image_lps[:, :, z].T, cmap="gray")
    for arr, (_, color, lw) in zip(masks_lps, CONTOUR_STYLES):
        for contour in measure.find_contours(arr[:, :, z].T.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)
    ax.axis("off")


def legend_handles():
    return [plt.Line2D([], [], color=c, label=l) for l, c, _ in CONTOUR_STYLES]


if __name__ == "__main__":
    # 1. 检查 NVIDIA 显卡加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的计算设备: {device.type.upper()}")

    # 2. 输入的 T2W 轴位前列腺 MRI（.nii.gz）
    mri_path = "601_MR_AXT2.nii.gz" # 8_MR_t2_tse_trasfov.nii.gz
    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"未找到 MRI 数据: {mri_path}")
    # 输出文件以输入文件名为前缀，例如 601_MR_AXT2.nii.gz -> 601_MR_AXT2_cg_mask.nii.gz
    out_prefix = os.path.basename(mri_path).removesuffix(".gz").removesuffix(".nii")
    # 导出的 .nii.gz 掩膜统一放到 output 文件夹
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 3~7. 载入模型并分割
    net = load_model(device)
    orig_img, mask, cg, pz, zones = segment(mri_path, net, device)

    voxel_ml = voxel_volume_ml(orig_img)
    print(f"分割完成！前列腺体积约 {mask.sum() * voxel_ml:.1f} mL "
          f"(CG {cg.sum() * voxel_ml:.1f} mL, PZ {pz.sum() * voxel_ml:.1f} mL)")

    # 8. 保存掩膜（与原始 MRI 相同的空间和 affine，可在 ITK-SNAP / 3D Slicer 中叠加查看）
    for name, arr in [("prostate_mask", mask), ("cg_mask", cg), ("pz_mask", pz), ("zones", zones)]:
        out_path = os.path.join(output_dir, f"{out_prefix}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr, orig_img.affine), out_path)
        print(f"掩膜已保存至: {out_path}")

    # 9. 可视化：在包含前列腺的轴位切片上绘制整个前列腺、CG、PZ 的边界轮廓
    image_lps, mask_lps, cg_lps, pz_lps = to_lps(orig_img, orig_img.get_fdata(), mask, cg, pz)
    slices = np.where(mask_lps.any(axis=(0, 1)))[0]
    if len(slices) == 0:
        print("未分割到前列腺，请检查输入图像是否为 T2W 轴位前列腺 MRI")
    else:
        show = np.unique(np.linspace(slices[0], slices[-1], min(6, len(slices))).round().astype(int))
        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        for ax in axes.flat:
            ax.axis("off")
        for ax, z in zip(axes.flat, show):
            draw_slice(ax, image_lps, (mask_lps, cg_lps, pz_lps), z)
            ax.set_title(f"Slice {z}")
        fig.legend(handles=legend_handles(), loc="lower center", ncol=3)
        fig.suptitle("MONAI Prostate / CG / PZ Boundary (T2W)")
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.savefig(f"{out_prefix}_prostate_boundary.png", dpi=150)
        plt.show()
