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

# 1. 检查 NVIDIA 显卡加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的计算设备: {device.type.upper()}")

# 2. 输入的 T2W 轴位前列腺 MRI（.nii.gz）
mri_path = "601_MR_AXT2.nii.gz"
if not os.path.exists(mri_path):
    raise FileNotFoundError(f"未找到 MRI 数据: {mri_path}")

# 3. 从 Hugging Face 的 MONAI 官方仓库下载前列腺解剖分割 Bundle 模型
# 该模型输出 背景(0) / 中央腺体CG(1) / 外周带PZ(2)，这里将 CG+PZ 合并为整个前列腺
model_name = "prostate_mri_anatomy"
model_dir = "./models"
bundle_root = os.path.join(model_dir, model_name)
if not os.path.exists(os.path.join(bundle_root, "models", "model.pt")):
    print("正在从 Hugging Face 下载 MONAI 官方前列腺分割 Bundle...")
    download(name=model_name, bundle_dir=model_dir)

# 4. 预处理流水线：必须与 Bundle 训练时一致（见 configs/inference.json 与 docs/README.md）
# 训练数据在轴位平面每侧中心裁剪 20%（center_crop.py 的默认 margin=0.2）
orig_img = nib.load(mri_path)
orig_shape = orig_img.shape[:3]
crop_margin = 0.2
preprocessing = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    CenterSpatialCropd(keys="image", roi_size=(
        int(orig_shape[0] * (1 - 2 * crop_margin)),
        int(orig_shape[1] * (1 - 2 * crop_margin)),
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

# 5. 按 Bundle 的 network_def 构建 UNet 并载入预训练权重
print("正在载入前列腺 3D 分割网络权重...")
net = UNet(
    spatial_dims=3, in_channels=1, out_channels=3,
    channels=(16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2),
    num_res_units=4, norm="batch", act="prelu", dropout=0.15,
).to(device)
state = torch.load(os.path.join(bundle_root, "models", "model.pt"), map_location=device)
net.load_state_dict(state.get("model", state))
net.eval()

# 6. 显卡加速推理
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

# 7. 整个前列腺的概率 = 1 - 背景概率，反变换回原始图像空间后再二值化，边界更平滑
data["pred"] = 1.0 - probs[0, :1].cpu()
data = Invertd(keys="pred", transform=preprocessing, orig_keys="image", nearest_interp=False)(data)
mask = (data["pred"] > 0.5).to(torch.uint8)
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

voxel_ml = np.prod(orig_img.header.get_zooms()[:3]) / 1000.0
print(f"分割完成！前列腺体积约 {mask.sum() * voxel_ml:.1f} mL")

# 8. 保存前列腺掩膜（与原始 MRI 相同的空间和 affine，可在 ITK-SNAP / 3D Slicer 中叠加查看）
out_mask_path = "601_MR_AXT2_prostate_mask.nii.gz"
nib.save(nib.Nifti1Image(mask, orig_img.affine), out_mask_path)
print(f"前列腺掩膜已保存至: {out_mask_path}")

# 9. 可视化：在包含前列腺的轴位切片上绘制前列腺边界轮廓
# 统一转到 LPS 方向，使 slice.T 显示为放射学视图（前方朝上、患者右侧在图像左侧）
to_lps = nib.orientations.ornt_transform(
    nib.orientations.io_orientation(orig_img.affine), nib.orientations.axcodes2ornt(("L", "P", "S")))
image_lps = nib.orientations.apply_orientation(orig_img.get_fdata(), to_lps)
mask_lps = nib.orientations.apply_orientation(mask, to_lps)

slices = np.where(mask_lps.any(axis=(0, 1)))[0]
if len(slices) == 0:
    print("未分割到前列腺，请检查输入图像是否为 T2W 轴位前列腺 MRI")
else:
    show = np.unique(np.linspace(slices[0], slices[-1], min(6, len(slices))).round().astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, z in zip(axes.flat, show):
        ax.imshow(image_lps[:, :, z].T, cmap="gray")
        for contour in measure.find_contours(mask_lps[:, :, z].T.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.5)
        ax.set_title(f"Slice {z}")
    fig.suptitle("MONAI Prostate Boundary (T2W)")
    plt.tight_layout()
    plt.savefig("601_MR_AXT2_prostate_boundary.png", dpi=150)
    plt.show()
