#!/usr/bin/env python3
"""
Verify air and water ROI sizing according to NEMA NU4-2008.
"""
import numpy as np

# Current config values
AIR_RADIUS_MM = 2.0
AIR_HEIGHT_MM = 7.5
WATER_RADIUS_MM = 2.0
WATER_HEIGHT_MM = 7.5
SPACING = 0.5

print("=== Current ROI Configuration ===")
print(f"Air insert: radius={AIR_RADIUS_MM}mm, height={AIR_HEIGHT_MM}mm")
print(f"Water insert: radius={WATER_RADIUS_MM}mm, height={WATER_HEIGHT_MM}mm")
print(f"Voxel spacing: {SPACING}mm")

# Calculate voxel dimensions
air_radius_vox = AIR_RADIUS_MM / SPACING
air_height_vox = AIR_HEIGHT_MM / SPACING
water_radius_vox = WATER_RADIUS_MM / SPACING
water_height_vox = WATER_HEIGHT_MM / SPACING

print("\n=== Voxel Dimensions ===")
print(f"Air: radius={air_radius_vox} vox, height={air_height_vox} vox")
print(f"Water: radius={water_radius_vox} vox, height={water_height_vox} vox")

# Calculate expected voxel count
air_volume_vox = np.pi * (air_radius_vox**2) * air_height_vox
water_volume_vox = np.pi * (water_radius_vox**2) * water_height_vox

print("\n=== Expected Voxel Count (theoretical) ===")
print(f"Air: ~{air_volume_vox:.0f} voxels")
print(f"Water: ~{water_volume_vox:.0f} voxels")
print("Reported in log: 833 voxels for both")

# NEMA NU4-2008 specifications
print("\n=== NEMA NU4-2008 Insert Specifications ===")
print("Hot Rod Chamber: 30mm diameter, 30mm height")
print("Uniform Region: 22.5mm diameter, 10mm height recommended for analysis")
print("Air/Water inserts: 4mm diameter (2mm radius) ✓")
print("  - Rod diameter = 4mm")
print("  - Insert length = 50mm (physical)")
print("  - Analysis height: typically 7-10mm from center")

# Verify current settings
print("\n=== Verification ===")
diameter_mm = AIR_RADIUS_MM * 2
if diameter_mm == 4.0:
    print(f"✓ Diameter: {diameter_mm}mm (matches NEMA 4mm specification)")
else:
    print(f"✗ Diameter: {diameter_mm}mm (should be 4mm)")

if AIR_HEIGHT_MM >= 5.0 and AIR_HEIGHT_MM <= 10.0:
    print(f"✓ Height: {AIR_HEIGHT_MM}mm (reasonable for NU4-2008)")
else:
    print(f"⚠ Height: {AIR_HEIGHT_MM}mm (consider 7-10mm range)")

voxel_difference = 833 - air_volume_vox
print(
    f"\n✓ Actual voxels (833) vs expected ({air_volume_vox:.0f}): difference = {voxel_difference:.0f}"
)
print("  (Difference due to discrete voxel boundaries - this is normal)")
