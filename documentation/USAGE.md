# NEMA Analysis Tool - Usage Guide

## nema_quant

### Excecution

To excecute `nema_quant`, provide the necesary paths and configurations settings through command line arguments:

- `input_image`: Nifti file to analize
- `--output`: Path to save the results
- `--config`: Path to custom YAML configuration file
- `--save-visualizations`: flag to activate save visualization mode
- `--visualizations-dir`: Directory to save visualization image (default: visualizations)
- `--advanced-metrics`: Calculated advanced metrics. **Must provide gt image**
- `--gt-image`:  Path to ground truth NIfTI image file for advanced metrics
- `--verbose`: Enable verbose output

Example of simple run:

```bash
nema_quant path/to/nitftii_file.nii \
--output path/to/output/file.txt \
--config path/to/config.yaml
```

Example of advance run:

```bash
nema_quant path/to/nitftii_file.nii \
--output path/to/output/file.txt \
--config path/to/config.yaml \
--advanced-metrics \
--gt-image path/to/niftii_file.nii \
--save_visualizations --verbose
```

## nema_quant_iter

### Excecution

To excecute `nema_quant_iter`, provide the necesary paths and configurations settings through command line arguments:

- `input_path`: Path for the input of the files iterations
- `--output`: Path to output files for results
- `--config`: Path to custom YAML configuyration file
- `--save-visualizations`: flag to activate save visualization mode
- `--visualizations-dir`: Directory to save visualization image (default: visualizations)
- `log_level`: Set logging level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR 50=CRITICAL (default: 20)
- `spacing`:  Voxel spacing in mm (x, y, z)
- `--verbose`: Enable verbose output

Example of simple run:

```bash
 nema_quant_iter path/to/directory/for/iterations/
--output path/to/output/file.txt \
 --config path/to/config.yaml
```

Example of advanced run:

```bash
 nema_quant_iter path/to/directory/for/iterations/ \
 --output path/to/output/file.txt \
 --config path/to/config.yaml \
 --save_visualizations --verbose
```

## nema_merge

- `xml_config`: Path to XML configuration file with experiments definitions
- `--output`: Output directory for merged analysis plots

### XML configuration file

```xml
<?xml version="1.0" encoding="UTF-8"?>
<experiments>
        <experiment
        name="test_name"
        path="/path/to/test/csv/analysis_results.csv"
        lung_path="/path/to/test/csv/lung_results.csv"
        dose="254.8571429"
        plot_status="grayed"/>
</experiments>
```

>
> - Both `path` and `lung_path` comes from the results of running individual test.
> - `dose` is present, will generate doseage analisis
> - `plot_status` gives you the ability to `grayout` or `enhanced` your results
> - If you activate advanced metrics and want to merge you need to pass  `advanced_path` in the xml file

## nema_coord

- `mm2vox,vox2mm`: Convertion command, used to transform the mm to voxel to get the coordinates to define the centers.
- `--dims`: dimmensions of your image
- `--spacing`: the spacing to use for the conversion

Example of simple run a mm2vox:

```bash
nema_coord mm2vox 58.84 23.74 -30.97 --dims 391 391 346 --spacing 2.0644 2.0644 2.0644
```

Example of simple run a vox2mm:

```bash
nema_cord vox2mm 158 207 158 --dims 391 391 346 --spacing 2.0644 2.0644 2.0644
```

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
nema_quant --help
```
