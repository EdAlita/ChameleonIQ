#!/usr/bin/env python3
"""
Verify uniform region mask positioning and coverage.
"""
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml  # type: ignore[import-untyped]


def create_cylindrical_mask(
    shape_zyx,
    center_zyx,
    radius_mm,
    height_mm,
    spacing_xyz,
):
    """Create a boolean mask for a cylinder aligned with the z-axis."""
    center_z, center_y, center_x = center_zyx
    radius_vox_x = radius_mm / spacing_xyz[0]
    radius_vox_y = radius_mm / spacing_xyz[1]
    half_height_vox_z = (height_mm / spacing_xyz[2]) / 2.0

    z_min = max(0, int(np.floor(center_z - half_height_vox_z)))
    z_max = min(shape_zyx[0] - 1, int(np.ceil(center_z + half_height_vox_z)))

    yy, xx = np.ogrid[: shape_zyx[1], : shape_zyx[2]]
    ellipse = ((xx - center_x) / radius_vox_x) ** 2 + (
        (yy - center_y) / radius_vox_y
    ) ** 2 <= 1.0

    mask = np.zeros(shape_zyx, dtype=bool)
    mask[z_min : z_max + 1, :, :] = ellipse

    return mask, z_min, z_max


# Load image
sitk_image = sitk.ReadImage("data/IQ_f50_Q_nosct 1.imgrec.nii")
data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

print(f"Image shape: {data.shape}")

# Load config
with open("src/config/nema_phantom_config_nu4_2008.yaml", "r") as f:
    config = yaml.safe_load(f)

SPACING = config["ROIS"]["SPACING"]
CENTRAL_SLICE = config["ROIS"]["CENTRAL_SLICE"]
ORIENTATION_Z = config["ROIS"]["ORIENTATION_Z"]
ORIENTATION_YX = config["ROIS"]["ORIENTATION_YX"]

# Get defaults for uniform region (from defaults.py)
UNIFORM_OFFSET_MM = 2.5  # Default value
UNIFORM_RADIUS_MM = 11.5  # Default value
UNIFORM_HEIGHT_MM = 10.0  # Default value

print("\n=== Configuration ===")
print(f"SPACING: {SPACING} mm")
print(f"CENTRAL_SLICE: {CENTRAL_SLICE}")
print(f"UNIFORM_OFFSET_MM: {UNIFORM_OFFSET_MM} mm")
print(f"UNIFORM_RADIUS_MM: {UNIFORM_RADIUS_MM} mm")
print(f"UNIFORM_HEIGHT_MM: {UNIFORM_HEIGHT_MM} mm")

threshold = np.max(data) * 0.41
binary_mask = data > threshold

# Calculate center of mass
weighted_slices = []
for z in range(data.shape[0]):
    slice_data = data[z]
    if np.sum(slice_data > threshold) > 0:
        M = np.sum(slice_data[slice_data > threshold])
        y_coords, x_coords = np.where(slice_data > threshold)
        cy = np.sum(y_coords * slice_data[y_coords, x_coords]) / M
        cx = np.sum(x_coords * slice_data[y_coords, x_coords]) / M
        weighted_slices.append((z, cy, cx, M))

if weighted_slices:
    total_mass = sum([m for _, _, _, m in weighted_slices])
    ce_z = sum([z * m for z, _, _, m in weighted_slices]) / total_mass
    ce_y = sum([cy * m for _, cy, _, m in weighted_slices]) / total_mass
    ce_x = sum([cx * m for _, _, cx, m in weighted_slices]) / total_mass

phantom_center_z = int(ce_z)
phantom_center_y = int(ce_y)
phantom_center_x = int(ce_x)

print("\n=== Detected Phantom Center ===")
print(f"Center (z, y, x): ({phantom_center_z}, {phantom_center_y}, {phantom_center_x})")

# Create uniform region mask (using detected center, not CENTRAL_SLICE)
uniform_center_z = phantom_center_z - ORIENTATION_Z * (UNIFORM_OFFSET_MM / SPACING)

print("\n=== Uniform Region Positioning ===")
print(f"Phantom center z: {phantom_center_z}")
print(
    f"Offset: {ORIENTATION_Z} * ({UNIFORM_OFFSET_MM} / {SPACING}) = {ORIENTATION_Z * (UNIFORM_OFFSET_MM / SPACING):.1f} voxels"
)
print(f"Uniform region center z: {uniform_center_z:.1f}")

uniform_mask, z_min, z_max = create_cylindrical_mask(
    shape_zyx=data.shape,
    center_zyx=(uniform_center_z, phantom_center_y, phantom_center_x),
    radius_mm=UNIFORM_RADIUS_MM,
    height_mm=UNIFORM_HEIGHT_MM,
    spacing_xyz=(SPACING, SPACING, SPACING),
)

print(f"Uniform mask z-range: {z_min} to {z_max} ({z_max - z_min + 1} slices)")
print(f"Total voxels in mask: {np.sum(uniform_mask)}")

# Calculate statistics
uniform_values = data[uniform_mask]
print("\n=== Uniform Region Statistics ===")
print(f"Mean: {np.mean(uniform_values):.6f}")
print(f"Std: {np.std(uniform_values):.6f}")
print(f"%STD: {100 * np.std(uniform_values) / np.mean(uniform_values):.2f}%")

# Visualize
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Show multiple z-slices
z_slices = [
    z_min,
    (z_min + z_max) // 2,
    z_max,
    phantom_center_z - 10,
    phantom_center_z,
    phantom_center_z + 10,
    CENTRAL_SLICE - 10,
    CENTRAL_SLICE,
    CENTRAL_SLICE + 10,
    70,
    80,
    90,
]

for idx, z in enumerate(z_slices):
    ax = axes[idx // 4, idx % 4]
    if z < data.shape[0]:
        # Show image
        ax.imshow(data[z], cmap="gray", vmin=0, vmax=np.percentile(data, 99))

        # Overlay mask
        mask_slice = uniform_mask[z]
        masked = np.ma.masked_where(~mask_slice, mask_slice)
        ax.imshow(masked, cmap="Reds", alpha=0.5)

        # Mark center
        ax.plot(
            phantom_center_x, phantom_center_y, "g+", markersize=12, markeredgewidth=2
        )

        title = f"z={z}"
        if z == z_min:
            title += " (uniform lower)"
        elif z == z_max:
            title += " (uniform upper)"
        elif z == phantom_center_z:
            title += " (phantom center)"
        elif z == CENTRAL_SLICE:
            title += " (hot rods)"

        ax.set_title(title)
    ax.axis("off")

plt.suptitle(
    f"Uniform Region Mask Verification\nMask: z={z_min}-{z_max}, center=({uniform_center_z:.1f}, {phantom_center_y}, {phantom_center_x})",
    fontsize=14,
)
plt.tight_layout()
plt.savefig("data/png/uniform_mask_verification.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved to: data/png/uniform_mask_verification.png")

# Create sagittal view
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Sagittal slice through center
sagittal = data[:, phantom_center_y, :]
ax.imshow(
    sagittal.T,
    cmap="gray",
    vmin=0,
    vmax=np.percentile(data, 99),
    aspect="auto",
    origin="lower",
)

# Overlay mask in sagittal view
mask_sagittal = uniform_mask[:, phantom_center_y, :]
masked_sag = np.ma.masked_where(~mask_sagittal, mask_sagittal)
ax.imshow(masked_sag.T, cmap="Reds", alpha=0.5, aspect="auto", origin="lower")

# Mark boundaries
ax.axvline(
    x=z_min, color="red", linestyle="--", linewidth=2, label=f"Uniform z={z_min}"
)
ax.axvline(
    x=z_max, color="red", linestyle="--", linewidth=2, label=f"Uniform z={z_max}"
)
ax.axvline(
    x=phantom_center_z,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Phantom center z={phantom_center_z}",
)
ax.axvline(
    x=CENTRAL_SLICE,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Hot rods z={CENTRAL_SLICE}",
)
ax.axvline(x=CENTRAL_SLICE - 10, color="orange", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=CENTRAL_SLICE + 10, color="orange", linestyle=":", linewidth=1, alpha=0.7)

ax.set_xlabel("Z slice", fontsize=12)
ax.set_ylabel("X coordinate", fontsize=12)
ax.set_title(f"Sagittal view (y={phantom_center_y}) - Uniform Region Mask", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/png/uniform_mask_sagittal.png", dpi=150, bbox_inches="tight")
print("Sagittal view saved to: data/png/uniform_mask_sagittal.png")

# Z-profile comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# Profile at phantom center
z_profile = [data[z, phantom_center_y, phantom_center_x] for z in range(data.shape[0])]
ax.plot(
    range(len(z_profile)), z_profile, "b-", linewidth=2, label="Intensity at center"
)

# Mark regions
ax.axvspan(
    z_min, z_max, alpha=0.3, color="red", label=f"Uniform region (z={z_min}-{z_max})"
)
ax.axvspan(
    CENTRAL_SLICE - 10,
    CENTRAL_SLICE + 10,
    alpha=0.3,
    color="orange",
    label=f"Hot rods range (z={CENTRAL_SLICE}±10)",
)

ax.axvline(
    x=phantom_center_z,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Phantom center z={phantom_center_z}",
)
ax.axvline(
    x=CENTRAL_SLICE,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Hot rods center z={CENTRAL_SLICE}",
)

ax.set_xlabel("Z slice", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Z-profile with Uniform Region and Hot Rods ranges", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/png/uniform_region_profile.png", dpi=150, bbox_inches="tight")
print("Z-profile saved to: data/png/uniform_region_profile.png")

print("\n=== Summary ===")
print(f"✓ Uniform region positioned at z={z_min}-{z_max}")
print(f"✓ Hot rods region at z={CENTRAL_SLICE - 10}-{CENTRAL_SLICE + 10}")
print(
    f"✓ Separation: {CENTRAL_SLICE - 10 - z_max} slices = {(CENTRAL_SLICE - 10 - z_max) * SPACING:.1f} mm"
)
if (CENTRAL_SLICE - 10 - z_max) > 0:
    print("✓ No overlap - regions are properly separated")
else:
    print("⚠ WARNING: Regions overlap or are too close!")

plt.show()
