# %%
from pathlib import Path

import ants
from aind_mri_utils.file_io.slicer_files import markup_json_to_dict

from aind_registration_utils.recipes import (
    individual_to_template_with_points_files,
)

# %%
template_dir = Path("/mnt/Data/MRI/templates")
processed_dir = Path(
    "/mnt/aind1-vast/scratch/ephys/persist/data/MRI/processed"
)
template_path = (
    template_dir / "template_15brain_n4_and_padding_cc_symetric.nii.gz"
)
template_targets = (
    template_dir / "fiducials_15brain_uw_template_2024_09_10.mrk.json"
)
# Mouse ID etc
mouse_id = "760332"
individual_dir = processed_dir / f"{mouse_id}"
individual_scan = individual_dir / f"{mouse_id}_100.nii.gz"
individual_brain_mask = individual_dir / f"{mouse_id}_auto_skull_strip.nrrd"

save_dir = Path("/home/galen.lynch/")

syn_kwargs = dict()
# %%
individual_to_template_with_points_files(
    individual_scan,
    individual_brain_mask,
    template_path,
    template_targets,
    save_dir=save_dir,
    mouse_id=mouse_id,
    syn_kwargs=syn_kwargs,
)

# %%
# %matplotlib ipympl

# %%
# Load the template image and the template target points
mouse_img = ants.image_read(str(individual_scan))
mouse_img_mask = ants.image_read(str(individual_brain_mask))
mouse_img_masked = mouse_img * mouse_img_mask

template_img = ants.image_read(str(template_path))
template_target_pts, _ = markup_json_to_dict(str(template_targets))

# %%
# debugging
fixed = template_img
moving = mouse_img_masked
syn_kwargs = dict(
    syn_sampling=2,
    reg_iterations=(500, 500, 500),
    syn_metric="CC",
    verbose=True,
)
rigid_kwargs = dict(aff_smoothing_sigmas=[3, 2, 1, 0], verbose=True)
affine_kwargs = dict(
    aff_smoothing_sigmas=[3, 2, 1, 0],
    verbose=True,
)
tx_rigid = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="Rigid",
    **rigid_kwargs,
)
# %%
tx_affine = ants.registration(
    fixed=fixed,
    moving=moving,
    initial_transform=tx_rigid["fwdtransforms"][0],
    type_of_transform="Affine",
    **affine_kwargs,
)
# %%
tx_syn = ants.registration(
    fixed=fixed,
    moving=moving,
    initial_transform=tx_affine["fwdtransforms"][0],
    type_of_transform="SyN",
    **syn_kwargs,
)

# %%
