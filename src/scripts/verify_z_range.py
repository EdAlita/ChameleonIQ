#!/usr/bin/env python3
"""
Verify z-range for hot rods to ensure no contamination from uniform region.
"""
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Load image using SimpleITK (same as nema_quant)
sitk_image = sitk.ReadImage(
    "data/IQ.05022024.DOI.petfus.3420s.att_no.filt.frame150.imgrec.nii"
)
data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

print(f"Image shape: {data.shape}")
print(f"Image range: [{data.min():.2f}, {data.max():.2f}]")

# Check intensity profile along z-axis at hot rods center
center_y, center_x = 75, 76  # From detected center
z_profile = [data[z, center_y, center_x] for z in range(data.shape[0])]

# Hot rods coordinates
hot_rods_coords = [
    (67, 87, "5mm"),
    (83, 88, "4mm"),
    (89, 73, "3mm"),
    (77, 62, "2mm"),
    (63, 71, "1mm"),
]

# Check mean intensity at different z levels for each rod
print("\n=== Intensity at hot rods positions vs z-level ===")
for y, x, name in hot_rods_coords:
    print(f"\n{name} rod at (y={y}, x={x}):")
    for z in [70, 75, 80, 85, 90, 95, 100, 105, 110]:
        if z < data.shape[0]:
            intensity = data[z, y, x]
            print(f"  z={z:3d}: {intensity:.3f}")

# Plot axial slices
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
z_slices = [70, 75, 80, 85, 90, 95, 100, 105, 110]

for idx, z in enumerate(z_slices):
    ax = axes[idx // 5, idx % 5]
    if z < data.shape[0]:
        ax.imshow(data[z], cmap="gray", vmin=0, vmax=np.percentile(data, 99))
        ax.set_title(f"z={z}")
        # Mark hot rods positions
        for y, x, _name in hot_rods_coords:
            ax.plot(x, y, "r+", markersize=8, markeredgewidth=1)
    ax.axis("off")

plt.suptitle("Axial slices - Hot rods range verification", fontsize=14)
plt.tight_layout()
plt.savefig("data/png/z_range_verification.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved to: data/png/z_range_verification.png")

# Plot z-profile at phantom center
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(range(len(z_profile)), z_profile, "b-", linewidth=2)
ax.axvline(x=90, color="r", linestyle="--", linewidth=2, label="CENTRAL_SLICE=90")
ax.axvline(
    x=80, color="orange", linestyle=":", linewidth=1.5, label="z=80 (lower bound)"
)
ax.axvline(
    x=100, color="orange", linestyle=":", linewidth=1.5, label="z=100 (upper bound)"
)
ax.axvline(
    x=48,
    color="g",
    linestyle="--",
    linewidth=2,
    label="Detected center z=48 (uniform region)",
)
ax.set_xlabel("Z slice", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title(
    "Intensity profile along z-axis at phantom center (y=75, x=76)", fontsize=14
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/png/z_profile_verification.png", dpi=150, bbox_inches="tight")
print("Z-profile saved to: data/png/z_profile_verification.png")

plt.show()
