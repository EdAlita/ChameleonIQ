# How NEMA Analysis Tool Works: A Complete Guide

*Transform your medical imaging quality assessment with automated NEMA NU 2-2018 analysis*

---

## 📖 Table of Contents

**Quick Navigation:**
- [🎯 What You'll Learn](#-what-youll-learn)
- [⚙️ Configuration System](#️-configuration-system)
- [📋 Understanding the NEMA Phantom](#-understanding-the-nema-phantom)
- [🔧 Data Preparation Process](#-data-preparation-process)
- [🎯 Analysis Workflow](#-analysis-workflow)
- [🧮 NEMA Metric Calculations](#-nema-metric-calculations)
- [🎨 Visualization & Reporting](#-visualization--reporting)
- [🔬 Advanced Features](#-advanced-features)
- [💡 Pro Tips & Troubleshooting](#-pro-tips--troubleshooting)
- [🚀 Getting Started](#-getting-started)

---

## 🎯 What You'll Learn

By the end of this guide, you'll understand:
- How the NEMA Analysis Tool automatically detects and measures phantom spheres
- How configuration files control activity ratios, voxel spacing, and calculation parameters
- Why the tool uses 12 background regions and how they're positioned
- How the advanced 3D offset optimization finds the perfect sphere locations
- How NEMA metrics are calculated using your specific scanner parameters
- What visualizations and reports the tool generates
- What happens "under the hood" during each analysis step

[↑ Back to Table of Contents](#-table-of-contents)

---

## ⚙️ Configuration System

### Why Configuration Matters
Unlike rigid analysis tools, the NEMA Analysis Tool adapts to **your specific scanner** and **phantom setup**. Every calculation parameter can be customized through configuration files, ensuring accurate results regardless of your equipment or protocol.

### 🔧 Key Configurable Parameters

**Activity Concentrations:**
- `cfg.ACTIVITY.HOT`: Activity concentration in hot spheres (Bq/ml)
- `cfg.ACTIVITY.BACKGROUND`: Background activity concentration (Bq/ml)
- **Impact**: These values determine the theoretical activity ratio used in contrast calculations

**Geometric Parameters:**
- `cfg.ROIS.SPACING`: Voxel spacing in mm (typically 2.0644mm for many scanners)
- `cfg.ROIS.CENTRAL_SLICE`: Which slice contains the sphere centers
- **Impact**: Critical for accurate millimeter-to-voxel conversions

**Analysis Settings:**
- `cfg.ROIS.BACKGROUND_OFFSET_YX`: Positions of the 12 background regions
- Background sphere count and positioning strategy
- **Impact**: Affects background statistics and measurement reliability

### Real-World Flexibility Examples

**Different Scanner Types:**
- **High-resolution scanner** (1mm voxels): Set `SPACING: 1.0`
- **Standard clinical scanner** (2mm voxels): Set `SPACING: 2.0644`
- **Research scanner** (0.5mm voxels): Set `SPACING: 0.5`

**Different Activity Protocols:**
- **Standard 4:1 ratio**: `HOT: 40000`, `BACKGROUND: 10000` Bq/ml
- **High contrast 8:1**: `HOT: 80000`, `BACKGROUND: 10000` Bq/ml
- **Low dose protocol**: `HOT: 20000`, `BACKGROUND: 5000` Bq/ml

[↑ Back to Table of Contents](#-table-of-contents)

---

## 📋 Understanding the NEMA Phantom

### What is a NEMA Phantom?
Think of the NEMA phantom as a "test dummy" for your medical scanner. It's a cylindrical container filled with radioactive water that contains six hollow spheres of different sizes (10mm, 13mm, 17mm, 22mm, 28mm, and 37mm). These spheres are filled with a radioactive solution that's typically **4 times more concentrated** than the background water (but this ratio is configurable!).

**Why these specific sizes?** Each sphere size tests your scanner's ability to detect objects of different volumes - from tiny lesions (10mm) to large tumors (37mm).

### The Magic Numbers: Configurable Phantom Geometry
The tool's default configuration knows exactly where each sphere should be located, but you can adjust these positions if your phantom differs:

- **37mm sphere**: The "reference" sphere at the phantom center [144, 112] mm
- **28mm sphere**: Upper right position [144, 178] mm  
- **22mm sphere**: Left side [95, 144] mm
- **17mm sphere**: Right side [161, 144] mm
- **13mm sphere**: Upper left [128, 127] mm
- **10mm sphere**: Far right [128, 161] mm

**Customization Tip:** If your phantom has different sphere positions, simply update the coordinates in your configuration file!

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🔧 Data Preparation Process

### Step 1: Loading and Validating Your Image
When you feed a medical image (NIfTI format) to the tool, here's what happens behind the scenes:

**🔍 Image Inspection**
- The tool opens your 3D image file and checks its dimensions
- It extracts crucial information like voxel spacing from the image header
- **Configuration Override**: If voxel spacing isn't in the header, it uses `cfg.ROIS.SPACING`
- It validates that the image is actually 3D and contains the expected data types

**📏 Coordinate System Setup**
The tool creates a "translation system" between:
- **Real-world coordinates** (millimeters) - where spheres should be physically located
- **Image coordinates** (voxels/pixels) - where they appear in your digital image
- **Conversion Formula**: `voxel_position = mm_position / cfg.ROIS.SPACING`

*Think of it like having both a street address and GPS coordinates for the same location.*

### Step 2: Activity Ratio Validation
Before any calculations begin, the tool validates your activity settings:

**🔬 Activity Validation Checks**
```
if activity_bkg <= 0 or (activity_hot / activity_bkg) <= 1:
    raise ValueError("Activity ratio must be greater than 1")
```

**📊 Activity Ratio Calculation**
The tool calculates the theoretical activity ratio term:
`activity_ratio_term = (cfg.ACTIVITY.HOT / cfg.ACTIVITY.BACKGROUND) - 1.0`

This value becomes crucial for the NEMA percent contrast formula later!

### Step 3: Creating the Digital Phantom Map
The tool builds a complete digital blueprint of your phantom using your configuration:

**🎯 Sphere Location Mapping**
For each sphere, it calculates:
- Exact center position in your image using your voxel spacing
- Radius in voxel units (accounting for your image's resolution)
- Expected volume for validation

**📍 Background Region Planning** 
The tool strategically places 12 background measurement circles using `cfg.ROIS.BACKGROUND_OFFSET_YX`:

**Primary Ring (60mm from center):**
- 8 positions forming a square pattern: North, South, East, West, and four corners
- Coordinates: [-60,-60], [-60,0], [-60,+60], [0,-60], [0,+60], [+60,-60], [+60,0], [+60,+60]

**Secondary Ring (42mm from center):**
- 4 diagonal positions filling the gaps
- Coordinates: [-42,-42], [-42,+42], [+42,-42], [+42,+42]

*All coordinates are automatically converted to voxel units using your configured spacing!*

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🎯 Analysis Workflow

### Multi-Slice Strategy - Configuration-Driven Slice Selection
The tool uses your central slice configuration to determine analysis levels:

**🎂 The Layer Cake Approach**
Starting from `cfg.ROIS.CENTRAL_SLICE`, the tool calculates:
- **Central slice**: Your configured primary measurement level
- **±10mm slices**: Converted to voxel offsets using your spacing
- **±20mm slices**: Two more levels for comprehensive sampling

**Calculation Example:**
```
cm_in_z_vox = 10mm / cfg.ROIS.SPACING  # Convert 1cm to voxels
slice_positions = [
    central_slice,
    central_slice + cm_in_z_vox,    # +10mm
    central_slice - cm_in_z_vox,    # -10mm
    central_slice + 2*cm_in_z_vox,  # +20mm
    central_slice - 2*cm_in_z_vox   # -20mm
]
```

### Advanced 3D Sphere Optimization - Finding the Perfect Spot

Here's where the tool gets really smart! The `_calculate_hot_sphere_counts_offset_zxy` function performs a comprehensive 3D search to find the absolute best measurement location for each sphere.

**🔍 The Smart Search Strategy**
The tool doesn't trust that spheres are perfectly positioned. Real-world phantoms can have slight manufacturing variations, positioning errors, or image registration issues. So it conducts an intelligent search to find the optimal measurement location.

**🗺️ 3D Search Grid Pattern**
For each sphere, the tool tests **130 different positions** in a 3D grid:

**XY-Plane Search Pattern (25 positions per slice):**
The tool creates a 5×5 grid of test positions around each sphere's expected center:
- **Row 1**: (-2,+2), (-1,+2), (0,+2), (+1,+2), (+2,+2)
- **Row 2**: (-2,+1), (-1,+1), (0,+1), (+1,+1), (+2,+1)  
- **Row 3**: (-2,0), (-1,0), **(0,0)**, (+1,0), (+2,0)
- **Row 4**: (-2,-1), (-1,-1), (0,-1), (+1,-1), (+2,-1)
- **Row 5**: (-2,-2), (-1,-2), (0,-2), (+1,-2), (+2,-2)

*The center position (0,0) represents the originally expected sphere location.*

**Z-Axis Search Range (5 slice levels):**
```python
offsets_z = [-2, -1, 0, 1, 2]  # Voxel offsets from central slice
```
The search explores different axial levels:
- **-2 slices**: Two slices below the expected position
- **-1 slice**: One slice below
- **0 slice**: The expected central slice
- **+1 slice**: One slice above  
- **+2 slices**: Two slices above the expected position

**🏆 The Optimization Process**
For each of the 130 test positions, the tool:

1. **Calculates the test position**: `offset_center = (center_yx[0] + offset[0], center_yx[1] + offset[1])`
2. **Creates a precise circular ROI** using your configured sphere radius
3. **Extracts all voxel values** within that circular region
4. **Calculates the mean activity**: `mean_count = np.mean(current_slice[roi_mask])`
5. **Compares with current best**: `if mean_count > max_mean: max_mean = mean_count`
6. **Updates the champion position**: `best_offset_zyx = (dz, offset[0], offset[1])`

**Debug Output Example:**
```
Found the best average counts for hot_sphere_37mm with offset (-1, 1, 0): 15247.52
```
This means the 37mm sphere's optimal position was 1 voxel up in Z, 1 voxel right in Y, and centered in X.

### Background Analysis with Configuration Precision
The background analysis uses your configured offset positions and sphere sizes:

**🎯 Size-Matched Background Regions**
For each sphere diameter, the tool creates background ROIs of matching size:
- **10mm sphere**: Gets 10mm-radius background ROIs
- **37mm sphere**: Gets 37mm-radius background ROIs
- All using your configured `BACKGROUND_OFFSET_YX` positions

**📈 Statistical Aggregation Across Multiple Slices**
Using your multi-slice configuration, background stats are calculated across all analysis slices, providing robust statistics that account for 3D variations.

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🧮 NEMA Metric Calculations

### Percent Contrast - Configured Activity Ratios
The percent contrast calculation uses your specific activity configuration:

**🔢 The Mathematical Journey**
```python
activity_ratio_term = (cfg.ACTIVITY.HOT / cfg.ACTIVITY.BACKGROUND) - 1.0
percent_contrast = ((C_H / C_B) - 1.0) / activity_ratio_term * 100.0
```

**Configuration Impact Examples:**
- **4:1 ratio**: `activity_ratio_term = 3.0`, expects 100% contrast for perfect performance
- **8:1 ratio**: `activity_ratio_term = 7.0`, allows detection of subtle performance differences
- **2:1 ratio**: `activity_ratio_term = 1.0`, maximizes sensitivity to small contrast changes

### Background Variability - Size-Specific Calculations
Background variability is calculated using size-matched statistics:

**📏 The Size-Matched Calculation**
```python
# For each sphere size, use matching background ROI size
C_B = background_stats[sphere_diam_mm]["C_B"]
SD_B = background_stats[sphere_diam_mm]["SD_B"]
percent_variability = (SD_B / C_B) * 100.0
```

This ensures fair comparison - small spheres are compared against small background regions, large spheres against large background regions.

### Lung Insert Correction - Configurable Reference
The lung insert analysis uses your configuration for:
- **Reference activity**: Uses the 37mm sphere background measurement (`CB_37`)
- **ROI size**: Fixed 15mm radius as per NEMA standard
- **Sampling range**: Automatically detected lung insert region

**What the Numbers Mean:**
- **100%** = Perfect! Scanner shows exact 4:1 ratio
- **80%** = Good performance, slight underestimation  
- **60%** = Moderate performance, noticeable blurring
- **<50%** = Poor performance, significant resolution loss

**Understanding Image Noise:**
- **Low variability (2-5%)**: Smooth, high-quality images
- **Moderate variability (5-10%)**: Acceptable for most clinical uses
- **High variability (>15%)**: Noisy images that may affect diagnosis

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🎨 Visualization & Reporting

### Real-Time ROI Visualization
When `save_visualizations=True`, the tool generates detailed plots showing:

**🎯 Sphere ROI Visualizations**
For each optimized sphere position, you get three-panel plots showing:
1. **Original Image with ROI**: Shows the sphere in context with ROI overlay
2. **ROI Mask**: The exact pixels included in measurement
3. **Masked Region Only**: Just the measured region for quality assessment

Each visualization includes statistical information:
```
Mean: 15247.52, Std: 1205.34, Pixels: 892
Optimized offset: (-1, 1, 0)
```

**📍 Background ROI Visualizations**
The background visualization shows:
- All 12 background regions numbered and positioned
- The pivot point (reference sphere center)
- Cyan circles showing exact ROI locations
- Verification that regions don't overlap with hot spheres

### Professional PDF Report Generation
The tool creates comprehensive PDF reports containing:

**📊 Executive Summary**
- Overall phantom performance metrics
- Comparison with NEMA acceptance criteria
- Configuration settings used for analysis
- Image acquisition parameters

**📋 Detailed Results Tables**
For each sphere size:
- Percent contrast (Q_H) with acceptance ranges
- Background variability (N) with quality indicators
- Raw measurement values (C_H, C_B, SD_B)
- Optimized positioning offsets found
- Statistical confidence indicators

**📈 Professional Plots and Charts**
- **Contrast vs. Sphere Size**: Shows resolution characteristics
- **Background Variability Trends**: Demonstrates noise performance
- **Lung Insert Correction**: Attenuation correction validation
- **ROI Positioning Maps**: Visual verification of measurement locations

**🔧 Configuration Documentation**
- Complete record of all analysis parameters
- Activity concentrations and ratios used
- Voxel spacing and geometric settings
- Background positioning strategy
- Quality control thresholds applied

### Advanced Plotting Features

**📊 Multi-Dimensional Analysis Plots**
The reporting system generates several types of professional visualizations:

**Performance Trend Plots:**
- Show how contrast recovery varies with sphere size
- Demonstrate the relationship between sphere size and detectability
- Include acceptance criteria lines for easy interpretation

**Background Uniformity Maps:**
- Spatial distribution of background measurements
- Heat maps showing measurement consistency
- Identification of potential systematic biases

**Optimization Results Visualization:**
- 3D surface plots showing the optimization landscape
- Before/after comparisons of ROI positioning
- Confidence intervals for optimized measurements

**Configuration Impact Analysis:**
- Sensitivity analysis showing how different activity ratios affect results
- Voxel spacing impact on measurement precision
- Background positioning strategy validation

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🔬 Advanced Features

### Dynamic Parameter Adjustment
The tool's configuration system enables:

**🎛️ Scanner-Specific Optimization**
```yaml
# High-resolution research scanner
ROIS:
  SPACING: 1.0  # 1mm voxels
  CENTRAL_SLICE: 120  # Adjusted for different phantom positioning

ACTIVITY:
  HOT: 80000    # Higher activity for better statistics
  BACKGROUND: 10000

# Clinical scanner  
ROIS:
  SPACING: 2.0644  # Standard clinical voxels
  CENTRAL_SLICE: 89   # Standard positioning

ACTIVITY:
  HOT: 40000    # Standard clinical activity
  BACKGROUND: 10000
```

**🔄 Protocol Flexibility**
The same analysis code adapts to:
- Different phantom manufacturers (with position adjustments)
- Various activity preparation protocols
- Multiple scanner types and configurations
- Research vs. clinical imaging protocols

### Quality Control Through Configuration

**📏 Automatic Validation**
The tool uses your configuration to validate:
- Whether ROIs fit within image boundaries
- If activity ratios make physical sense
- Whether voxel spacing produces reasonable geometric conversions
- If background regions avoid hot sphere contamination

**🚨 Configuration-Based Warnings**
```python
if activity_bkg <= 0 or (activity_hot / activity_bkg) <= 1:
    raise ValueError("Activity ratio must be greater than 1")

# Warns if spheres extend beyond image boundaries
# Alerts if voxel spacing seems unrealistic
# Flags if background regions are too close to hot spheres
```

[↑ Back to Table of Contents](#-table-of-contents)

---

## 💡 Pro Tips & Troubleshooting

### Optimizing Your Configuration

**🎯 Activity Ratio Selection**
- **4:1 (Standard)**: Good for routine QA, follows NEMA recommendations
- **6:1 or 8:1**: Better sensitivity for detecting small performance changes
- **2:1**: Maximum sensitivity but may be affected by noise

**📏 Voxel Spacing Accuracy**
- Always verify voxel spacing matches your scanner's actual pixel size
- Use image header information when available
- Cross-check with phantom manufacturer specifications

**🎚️ Background Region Optimization**
- Standard offsets work for most phantoms
- Adjust if your phantom has unusual sphere arrangements
- Ensure adequate distance from all hot spheres

### Interpreting Configuration-Dependent Results

**📊 Understanding Activity Ratio Impact**
Higher activity ratios (8:1 vs 4:1) will:
- Show lower percent contrast values for the same image quality
- Provide better discrimination between different reconstruction algorithms
- Be more sensitive to partial volume effects

**📏 Voxel Spacing Considerations**
Smaller voxels (higher resolution) will:
- Provide more accurate ROI definitions
- Reduce partial volume effects
- May increase noise in individual measurements
- Enable detection of smaller positioning errors

### Understanding 3D Optimization Results
1. **Small Offsets (±1 voxel)**: Normal variation, indicates good phantom setup
2. **Medium Offsets (±2 voxels)**: Acceptable variation, may indicate minor positioning issues
3. **Large Offsets (>±2 voxels)**: May indicate phantom positioning problems or image registration issues
4. **Consistent Patterns**: If all spheres show similar offset directions, consider phantom alignment

### Troubleshooting Common Issues

**🔍 Common Configuration Issues**
1. **Wrong voxel spacing**: Results in incorrect ROI sizes and positions
2. **Incorrect activity ratio**: Leads to misleading contrast percentages  
3. **Poor central slice selection**: Affects all measurements systematically
4. **Background regions too close**: Contaminates background with hot sphere activity

**✅ Configuration Validation Checklist**
- [ ] Voxel spacing matches scanner specifications
- [ ] Activity ratio reflects actual phantom preparation
- [ ] Central slice contains sphere centers
- [ ] Background offsets maintain adequate separation
- [ ] All coordinates use consistent units (mm vs voxels)

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🚀 Getting Started

### Step-by-Step Setup Process

1. **📋 Gather Your Scanner Information**
   - Voxel spacing from scanner specifications
   - Typical central slice for your phantom setup
   - Activity concentrations used in phantom preparation

2. **⚙️ Create Your Configuration File**
   ```yaml
   ROIS:
     SPACING: 2.0644  # Your scanner's voxel spacing
     CENTRAL_SLICE: 89  # Adjust for your setup
   
   ACTIVITY:
     HOT: 40000      # Your hot sphere activity
     BACKGROUND: 10000  # Your background activity
   ```

3. **🧪 Test with Known Data**
   - Run analysis on a reference phantom scan
   - Verify ROI positioning with visualizations
   - Compare results with expected values

4. **📊 Validate and Refine**
   - Check debug output for optimization offsets
   - Review generated visualizations
   - Adjust configuration if needed

### Quick Start Commands

```bash
# Basic analysis
python -m nema_quant analyze input_image.nii

# With visualizations
python -m nema_quant analyze input_image.nii --save-visualizations

# With custom configuration
python -m nema_quant analyze input_image.nii --config my_config.yaml
```

**Ready to revolutionize your medical imaging quality assurance? The NEMA Analysis Tool's configurable approach ensures accurate results for any scanner, any protocol, any phantom setup!**

[↑ Back to Table of Contents](#-table-of-contents)

---

*This tool implements the NEMA NU 2-2018 standard with scientific rigor while providing unprecedented flexibility through comprehensive configuration management. Whether you're conducting routine quality assurance with standard protocols or advanced research with custom setups, the configurable analysis ensures consistent, reliable, and traceable results tailored to your specific requirements.*