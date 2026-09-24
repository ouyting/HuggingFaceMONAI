"""前列腺 MRI 分割网页应用（Gradio）。

上传 T2W 轴位前列腺 MRI 的 DICOM 序列（多个 .dcm 文件或一个 .zip 压缩包，也支持 .nrrd、.nii/.nii.gz），
调用 HF_monai_prostate_pipeline 中的 MONAI Bundle 模型分割出整个前列腺、中央腺体 CG 与外周带 PZ，
在网页上逐层浏览分割轮廓并下载掩膜。

运行: python prostate_seg_app.py  然后浏览器打开 http://127.0.0.1:7860
"""
import os
import shutil
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")  # 服务器端绘图，不弹出窗口
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import gradio as gr

from HF_monai_prostate_pipeline import (
    load_model, segment, voxel_volume_ml, to_lps, draw_slice, legend_handles, CONTOUR_STYLES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的计算设备: {device.type.upper()}")
net = load_model(device)  # 启动时载入一次，所有请求共用


VOLUME_EXTS = (".nii", ".nii.gz", ".nrrd")


def collect_files(uploaded, dicom_dir):
    """把上传的文件（含 zip 内的文件）平铺复制到 dicom_dir，返回其中的 NIfTI / NRRD 体数据路径（若有）。"""
    volume = None
    for i, path in enumerate(uploaded):
        name = os.path.basename(path)
        if name.lower().endswith(VOLUME_EXTS):
            volume = path
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for j, member in enumerate(zf.infolist()):
                    if member.is_dir():
                        continue
                    with zf.open(member) as src, open(os.path.join(dicom_dir, f"{i}_{j}"), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            shutil.copy(path, os.path.join(dicom_dir, f"{i}_{name}"))
    return volume


def dicom_to_nifti(dicom_dir, out_path):
    """读取目录中切片数最多的 DICOM 序列并转成 NIfTI，返回序列描述。"""
    reader = sitk.ImageSeriesReader()
    series = [(sid, reader.GetGDCMSeriesFileNames(dicom_dir, sid))
              for sid in reader.GetGDCMSeriesIDs(dicom_dir)]
    if not series:
        raise gr.Error("未在上传文件中找到有效的 DICOM 图像")
    sid, files = max(series, key=lambda s: len(s[1]))
    if len(files) < 10:
        raise gr.Error(f"DICOM 序列只有 {len(files)} 张切片，请上传完整的 3D T2W 轴位序列")
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    sitk.WriteImage(reader.Execute(), out_path)

    desc = reader.GetMetaData(0, "0008|103e").strip() if reader.HasMetaDataKey(0, "0008|103e") else "未命名序列"
    info = f"{desc}（{len(files)} 张切片）"
    if len(series) > 1:
        info += f"；上传中共有 {len(series)} 个序列，已自动选择切片数最多的一个"
    return info


def fig_to_array(fig):
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def render_slice(state, z, visible):
    if not state:
        return None
    z = int(z)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_slice(ax, state["image"], state["masks"], z, visible)
    ax.set_title(f"Slice {z}")
    if visible:
        ax.legend(handles=legend_handles(visible), loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig_to_array(fig)


def render_overview(state, visible):
    slices = state["slices"] if state else []
    if len(slices) == 0:
        return None
    show = np.unique(np.linspace(slices[0], slices[-1], min(6, len(slices))).round().astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, z in zip(axes.flat, show):
        draw_slice(ax, state["image"], state["masks"], z, visible)
        ax.set_title(f"Slice {z}")
    if visible:
        fig.legend(handles=legend_handles(visible), loc="lower center", ncol=3)
    fig.suptitle("MONAI Prostate / CG / PZ Boundary (T2W)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig_to_array(fig)


def run(uploaded, visible, progress=gr.Progress()):
    if not uploaded:
        raise gr.Error("请先上传 DICOM 文件或 zip 压缩包")
    uploaded = [f if isinstance(f, str) else f.name for f in uploaded]
    workdir = tempfile.mkdtemp(prefix="prostate_seg_")
    dicom_dir = os.path.join(workdir, "dicom")
    os.makedirs(dicom_dir)

    progress(0.1, desc="读取 DICOM...")
    mri_path = collect_files(uploaded, dicom_dir)
    if mri_path:
        series_info = os.path.basename(mri_path)
        if mri_path.lower().endswith(".nrrd"):
            # 分割流水线用 nibabel 读图，NRRD 先经 SimpleITK 转成 NIfTI（保留方向与间距）
            nifti_path = os.path.join(workdir, "input_t2w.nii.gz")
            sitk.WriteImage(sitk.ReadImage(mri_path), nifti_path)
            mri_path = nifti_path
    else:
        mri_path = os.path.join(workdir, "input_t2w.nii.gz")
        series_info = dicom_to_nifti(dicom_dir, mri_path)

    progress(0.3, desc="3D 分割推理中...")
    orig_img, mask, cg, pz, zones = segment(mri_path, net, device)

    progress(0.8, desc="生成结果...")
    out_files = []
    for name, arr in [("prostate_mask", mask), ("cg_mask", cg), ("pz_mask", pz), ("zones", zones)]:
        out_path = os.path.join(workdir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr, orig_img.affine), out_path)
        out_files.append(out_path)
    if mri_path.startswith(workdir):
        out_files.insert(0, mri_path)  # 附上 DICOM / NRRD 转换后的原图，方便在 3D Slicer / ITK-SNAP 中叠加查看

    voxel_ml = voxel_volume_ml(orig_img)
    zooms = orig_img.header.get_zooms()[:3]
    summary = (
        f"**序列**：{series_info}  \n"
        f"**图像尺寸**：{' × '.join(map(str, orig_img.shape[:3]))}，"
        f"体素间距 {' × '.join(f'{s:.2f}' for s in zooms)} mm  \n"
        f"**前列腺体积**：{mask.sum() * voxel_ml:.1f} mL"
        f"（CG {cg.sum() * voxel_ml:.1f} mL，PZ {pz.sum() * voxel_ml:.1f} mL）"
    )

    image_lps, mask_lps, cg_lps, pz_lps = to_lps(orig_img, orig_img.get_fdata(dtype=np.float32), mask, cg, pz)
    slices = np.where(mask_lps.any(axis=(0, 1)))[0]
    state = {"image": image_lps, "masks": (mask_lps, cg_lps, pz_lps), "slices": slices}
    n_slices = image_lps.shape[2]
    if len(slices) == 0:
        summary += "\n\n⚠️ 未分割到前列腺，请检查输入图像是否为 T2W 轴位前列腺 MRI"
        z0 = n_slices // 2
    else:
        z0 = int(slices[len(slices) // 2])

    slider = gr.Slider(minimum=0, maximum=n_slices - 1, value=z0, step=1, interactive=True)
    return (summary, render_overview(state, visible), slider, render_slice(state, z0, visible),
            out_files, state)


with gr.Blocks(title="前列腺 MRI 分割") as demo:
    gr.Markdown(
        "# 前列腺 MRI 分割（MONAI prostate_mri_anatomy）\n"
        "上传 **T2W 轴位** 前列腺 MRI 的 DICOM 序列：可多选同一序列的全部 `.dcm` 文件，或上传包含该序列的 `.zip` 压缩包"
        "（也支持 `.nrrd`、`.nii` / `.nii.gz`）。绿色为整个前列腺，红色为中央腺体 CG，蓝色为外周带 PZ。\n\n"
        "*仅供科研演示，不能用于临床诊断。*"
    )
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="上传 DICOM 文件 / zip / NRRD", file_count="multiple", type="filepath")
            run_btn = gr.Button("开始分割", variant="primary")
            structure_names = [name for name, _, _ in CONTOUR_STYLES]
            visible = gr.CheckboxGroup(structure_names, value=structure_names, label="显示的分割掩膜")
            summary = gr.Markdown()
            downloads = gr.File(label="下载分割掩膜 (NIfTI)", file_count="multiple", interactive=False)
        with gr.Column(scale=2):
            with gr.Tab("逐层浏览"):
                slice_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="轴位切片", interactive=False)
                slice_view = gr.Image(label="分割轮廓", type="numpy", interactive=False)
            with gr.Tab("概览"):
                overview = gr.Image(label="前列腺范围内均匀选取的 6 层切片", type="numpy", interactive=False)

    run_btn.click(run, inputs=[files, visible],
                  outputs=[summary, overview, slice_slider, slice_view, downloads, state])
    slice_slider.change(render_slice, inputs=[state, slice_slider, visible], outputs=slice_view,
                        show_progress="hidden")
    visible.change(lambda s, z, v: (render_slice(s, z, v), render_overview(s, v)),
                   inputs=[state, slice_slider, visible], outputs=[slice_view, overview], show_progress="hidden")


if __name__ == "__main__":
    demo.launch()
