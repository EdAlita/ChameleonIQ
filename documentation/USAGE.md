# NEMA Analysis Tool - Usage Guide

## Quick Start

### Basic Usage

```bash
# Analyze a NIfTI image with default settings
python main.py input.nii --output results.txt

# Or using the module
python -m nema_quant input.nii --output results.txt
```

### Advanced Usage

```bash
# Use custom configuration file
python main.py input.nii --config config/example_config.yaml --output results.txt

# Specify custom voxel spacing (if not in image header)
python main.py input.nii --spacing 2.0644 2.0644 2.0644 --output results.txt

# Skip registration step
python main.py input.nii --no-merge --output results.txt

# Verbose output for debugging
python main.py input.nii --output results.txt --verbose
```

## Command Line Options

- `input_image`: Path to input NIfTI file (.nii or .nii.gz) **[REQUIRED]**
- `--output, -o`: Path to output text file **[REQUIRED]**
- `--config, -c`: Path to custom YAML configuration file (optional)
- `--spacing`: Voxel spacing in mm (x y z) (optional, read from header if not provided)
- `--no-merge`: Skip image registration/merging step
- `--verbose, -v`: Enable verbose output
- `--version`: Show version information

## Configuration File

You can customize the analysis by providing a YAML configuration file. See `config/example_config.yaml` for an example.

### Key Configuration Parameters:

```yaml
# Activity concentrations
ACTIVITY:
  HOT: 0.79          # Hot sphere activity
  BACKGROUND: 0.079  # Background activity

# ROI parameters
ROIS:
  CENTRAL_SLICE: 172              # Central slice for analysis
  BACKGROUND_OFFSET_YX: [...]     # Background ROI positions

# Phantom definitions
PHANTHOM:
  ROI_DEFINITIONS_MM: [...]       # Sphere definitions
```

## Output Format

The tool generates a detailed text report containing:

- Analysis configuration
- Results table with NEMA metrics
- Legend and formulas
- Summary statistics

### Example Output:

```
================================================================================
NEMA NU 2-2018 IMAGE QUALITY ANALYSIS RESULTS
================================================================================
Generated on: 2025-07-16 10:30:00
Input image: /path/to/input.nii
Voxel spacing: 2.0644 x 2.0644 x 2.0644 mm

ANALYSIS CONFIGURATION:
----------------------------------------
Hot activity: 0.790
Background activity: 0.079
Activity ratio: 10.00
Central slice: 172

ANALYSIS RESULTS:
----------------------------------------
Sphere Analysis Results (NEMA NU 2-2018 Section 7.4.1)

Diameter   Q_H (%)    N (%)      C_H          C_B          SD_B
(mm)                             (counts)     (counts)     (counts)
----------------------------------------------------------------------------
37         85.23      2.45       1234.56      678.90       16.72
28         82.15      2.58       1198.43      672.11       17.34
...
```

## Examples

### Example 1: Basic Analysis
```bash
python main.py data/phantom_image.nii --output results/analysis.txt
```

### Example 2: Custom Configuration
```bash
python main.py data/phantom_image.nii \
    --config my_config.yaml \
    --output results/custom_analysis.txt \
    --verbose
```

### Example 3: Manual Spacing
```bash
python main.py data/phantom_image.nii \
    --spacing 1.5 1.5 2.0 \
    --output results/analysis_manual_spacing.txt
```

## Troubleshooting

### Common Issues:

1. **ImportError**: Make sure all dependencies are installed
   ```bash
   pip install -e .
   ```

2. **File not found**: Check that the input NIfTI file exists and path is correct

3. **Configuration errors**: Validate your YAML configuration file syntax

4. **Memory issues**: For large images, ensure sufficient RAM is available

5. **SimpleITK warnings**: These are harmless deprecation warnings and can be ignored

### Getting Help:

```bash
python main.py --help
```

For verbose debugging output:
```bash
python main.py input.nii --output results.txt --verbose
```
