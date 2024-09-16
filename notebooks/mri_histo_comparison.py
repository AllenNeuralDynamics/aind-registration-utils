# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ants
#     language: python
#     name: ants
# ---

# %%

import os
import psutil

nthread = psutil.cpu_count(logical=False)
thread_env_vars = [
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

for env_var in thread_env_vars:
    os.environ[env_var] = str(nthread)


# %%
import os

import ants

# import SimpleITK as sitk
from matplotlib import pyplot as plt
from pathlib import Path

from aind_registration_utils.ants import (
    apply_ants_transforms_to_point_dict,
    ants_register_syn,
    combine_syn_txs,
    combine_syn_and_second_transform,
)

# %matplotlib ipympl

# %%
base_data_path = Path("/mnt/data/local_data/")
processed_path = base_data_path / "mri" / "processed"
mri_template_path = (
    base_data_path
    / "mri"
    / "templates"
    / "template_15brain_n4_and_padding_cc_symetric.nii.gz"
)
template_fiducials_path = (
    base_data_path / "mri" / "templates" / "UWFiducials_V2Symetrical.fcsv"
)
results_path = Path("/mnt/data/analysis/sac-mri-targeting/ants_transforms")

ls_template_path = Path(
    "/mnt/data/local_data/light_sheet/templates/smartspim_lca_template_25.nii.gz"
)
ccf_template_path = Path(
    "/mnt/data/local_data/ccf/official/average_template_25.nii.gz"
)

annotated_mice = [717381, 721678, 721679, 721680, 721682, 721685, 743700]

processed_paths = {}
processed_paths[721680] = processed_path / "721680" / "UW"
processed_paths[721682] = processed_path / "721682" / "UW2"
processed_paths[721685] = processed_path / "721685" / "HF"
mri_target_paths = {}
for m in annotated_mice:
    processed_paths.setdefault(m, processed_path / str(m))
    mri_target_paths[m] = processed_paths[m] / f"{m}_ManualAnnotations.fcsv"

mask_paths = {}
mri_img_paths = {}
for m in [717381, 721680]:
    mask_paths[m] = processed_paths[m] / f"{m}_auto_skull_strip_1.nrrd"
for m in annotated_mice:
    mask_paths.setdefault(m, processed_paths[m] / f"{m}_auto_skull_strip.nrrd")
    mri_img_paths[m] = processed_paths[m] / f"{m}_100.nii.gz"


# %%
def ants_similarity_transform(
    fixed_img,
    moving_img,
    rigid_kwargs=dict(),
    similarity_kwargs=dict(
        aff_metric="mattes",
    ),
    save_prefix="",
):
    """
    Perform SyN registration using ANTs with a two-stage initialization (rigid
    followed by affine).

    This function performs registration of the moving image to the fixed image
    using the Symmetric Normalization (SyN) method implemented in ANTs
    (Advanced Normalization Tools).  Before the SyN registration, it employs a
    two-stage initialization approach: first, it computes a rigid
    transformation, followed by an affine transformation. The final SyN
    registration is initialized with the affine transformation.

    Parameters
    ----------
    fixed_img : ants.ANTsImage
        Target image for the registration.
    moving_img : ants.ANTsImage
        Source image that will be aligned to the `fixed_img`.
    syn_save_prefix : str, optional
        Prefix for the output files of the SyN registration. If not specified,
        no prefix is added.

    Returns
    -------
    dict
        A dictionary containing the results of the SyN registration, including
        forward and inverse transformations, warp fields, and other
        registration details.

    Examples
    --------
    >>> fixed = ants.image_read('path/to/fixed_image.nii.gz')
    >>> moving = ants.image_read('path/to/moving_image.nii.gz')
    >>> syn_results = ants_register_syn_cc(fixed, moving, 'output_prefix_')
    """

    tx_rigid = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform="Rigid",
        **rigid_kwargs,
    )
    tx_similarity = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        initial_transform=tx_rigid["fwdtransforms"][0],
        type_of_transform="Similarity",
        outprefix=str(save_prefix),
        **similarity_kwargs,
    )
    return tx_similarity


# %%
ccf_img = ants.image_read(str(ccf_template_path))
ls_img = ants.image_read(str(ls_template_path))

# %%
sim_tx = ants_similarity_transform(
    ccf_img,
    ls_img,
    save_prefix=results_path / "ls-ccf-sim-",
    rigid_kwargs=dict(
        verbose=True,
        aff_iterations=(1000, 500, 500, 10),
    ),
    similarity_kwargs=dict(verbose=True, aff_iterations=(1000, 500, 500, 10)),
)
# %%

mouse_img = ants.image_read(str(mouse_img_file))
mouse_img_mask = ants.image_read(str(mouse_mask_file))
mouse_img_masked = mouse_img * mouse_img_mask

warp_path = Path("/root/capsule/data/antsreg-mri-ccf-yoni-AMBMC-2023-10-04/")
save_path = Path("/root/capsule/results/")

target_landmarks_file = data_path / "landmark_annotations.npy"

in_vivo_path = (
    mri_template_path
    / "mri-invivo-yoni-uw-template-15brain-n4-and-padding-cc-symmetric/template_15brain_n4_and_padding_cc_symetric.nii.gz"
)
ccf_path = ccf_template_path / "average_template_25.nii.gz"
ccf_tgts_path = data_path / "ccf-centroid-targets.fcsv"

ccf_invivo_file = "ccf-invivo-comptx.nii.gz"
invivo_ccf_file = "invivo-ccf-comptx.nii.gz"

mouse = 679812
mouse_path = data_path / "{}_hutch/".format(mouse)
mouse_img_file = mouse_path / "{}_raw.nii.gz".format(mouse)
mouse_mask_file = mouse_path / "{}_manual_skull_strip.seg.nrrd".format(mouse)

mouse_save_path = save_path / str(mouse)

# %%
mri_template_path = Path(
    "/mnt/aind1-vast/scratch/ephys/persist/data/MRI/Templates/"
)
mri_transform_path = Path("/home/galen.lynch/Downloads/")
fiducial_paths = {
    "yoni": mri_template_path
    / "template_15brain_n4_and_padding_cc_symmetric/fiducials_template_15brain_n4_and_padding_cc_symmetric.fcsv",
    "qiu": mri_template_path / "qiu_template/fiducials_qiu_template.fcsv",
}
save_path = Path("/home/galen.lynch/")
transform_paths = {
    "yoni": mri_transform_path / "684810-invivo-comptx-yoni.nii.gz",
    "qiu": mri_transform_path / "684810-invivo-comptx-qiu.nii.gz",
}


# %%
def compare_planes(indexfun, axs, test, ref):
    axs[0].imshow(indexfun(test), cmap="gray")
    axs[1].imshow(indexfun(ref), cmap="gray")


# %%

for fiducial_name, fiducial_path in fiducial_paths.items():
    warp_path = transform_paths[fiducial_name]
    pts = read_slicer_fcsv(fiducial_path)
    warped_pts = apply_ants_transforms_to_point_dict(pts, [str(warp_path)])
    create_slicer_fcsv(
        save_path / "{}-warped.fcsv".format(fiducial_name),
        warped_pts,
        direction="LPS",
    )


# %%

# %%
ccf_img = ants.image_read(str(ccf_template_path))
ls_img = ants.image_read(str(ls_template_path))


# %%
mouse_img = ants.image_read(str(mouse_img_file))
mouse_img_mask = ants.image_read(str(mouse_mask_file))
mouse_img_masked = mouse_img * mouse_img_mask

# %%
tx_dict = dict()
for template_name, template_info in invivo_templates.items():
    # paths
    warp_path = template_info["warp-path"]
    ccf_invivo_path = warp_path / ccf_invivo_file
    invivo_ccf_path = warp_path / invivo_ccf_file
    mouse_template_save_path = mouse_save_path / (template_name + "/")
    mouse_invivo_prefix = mouse_template_save_path / "{}-invivo-".format(mouse)
    invivo_mouse_prefix = mouse_template_save_path / "invivo-{}-".format(mouse)
    mouse_ccf_prefix = mouse_template_save_path / "{}-ccf-".format(mouse)
    ccf_mouse_prefix = mouse_template_save_path / "ccf-{}-".format(mouse)
    mouse_target_savefile = (
        mouse_template_save_path / "targets-{}-transformed.fcsv".format(mouse)
    )

    if not os.path.isdir(mouse_template_save_path):
        mouse_template_save_path.mkdir(parents=True, exist_ok=True)

    invivo_img = ants.image_read(str(template_info["in-vivo-path"]))
    mouse_invivo_tx_syn = ants_register_syn_cc(
        invivo_img, mouse_img_masked, mouse_invivo_prefix
    )

    mouse_invivo_tx_cmb, invivo_mouse_tx_cmb = combine_mouse_invivo_txs(
        invivo_img,
        mouse_img_masked,
        mouse_invivo_tx_syn,
        mouse_invivo_prefix,
        invivo_mouse_prefix,
    )

    (
        mouse_ccf_tx_cmb,
        ccf_mouse_tx_cmb,
    ) = combine_mouse_invivo_and_invivo_ccf_txs(
        invivo_img,
        mouse_img_masked,
        mouse_invivo_tx_syn,
        invivo_ccf_path,
        ccf_invivo_path,
        mouse_ccf_prefix,
        ccf_mouse_prefix,
    )

    tgt_pts_mouse_lps = apply_ants_transforms_to_point_dict(
        tgts_ccf_coords, mouse_ccf_tx_cmb
    )
    create_slicer_fcsv(mouse_target_savefile, tgt_pts_mouse_lps)

    tx_dict[template_name] = {
        "mouse-invivo-tx-syn": mouse_invivo_tx_syn,
        "mouse-invivo-tx-cmb": mouse_invivo_tx_cmb,
        "invivo-mouse-tx-cmb": invivo_mouse_tx_cmb,
        "mouse-ccf-tx-cmb": mouse_ccf_tx_cmb,
        "ccf-mouse-tx-cmb": ccf_mouse_tx_cmb,
        "mouse-target-file": mouse_target_savefile,
    }

# %%
f, axs = plt.subplots(1, 2, figsize=(5, 2), sharex=True, sharey=True)
compare_planes(
    lambda x: x[:, :, 80], axs, invivo_img, mouse_invivo_tx_syn["warpedmovout"]
)
[ax.set_axis_off() for ax in axs]
f.tight_layout()


# %%
m_in_c_img = ants.apply_transforms(
    fixed=ccf_img,
    moving=mouse_img_masked,
    transformlist=mouse_ccf_tx_cmb,
)

f, axs = plt.subplots(1, 2, figsize=(5, 3), sharex=True, sharey=True)
compare_planes(lambda x: x[:, :, 250], axs, ccf_img, m_in_c_img)
(ax.axis_off for ax in axs)
f.tight_layout()

# %%
c_in_m_img = ants.apply_transforms(
    fixed=mouse_img_masked,
    moving=ccf_img,
    transformlist=ccf_mouse_tx_cmb,
)


# %%
f, axs = plt.subplots(1, 2, figsize=(5, 2), sharex=True, sharey=True)
compare_planes(lambda x: x[:, :, 45], axs, mouse_img_masked, c_in_m_img)
[ax.set_axis_off() for ax in axs]
f.tight_layout()
