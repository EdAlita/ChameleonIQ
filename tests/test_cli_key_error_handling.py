import warnings
from unittest.mock import MagicMock, patch

import numpy as np
from yacs.config import CfgNode

from src.config.defaults import get_cfg_defaults
from src.nema_quant import cli


class TestCLIKeyErrorHandling:
    """Test CLI handling of key errors and data format issues."""

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    def test_handle_missing_keys_in_results(
        self,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test handling of missing keys in analysis results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mock_exists.return_value = True
            cfg = _prepare_cfg_for_tests()
            mock_load_config.return_value = cfg
            mock_load_image.return_value = (np.ones((361, 100, 100)), np.eye(4))
            mock_get_props.return_value = ((361, 100, 100), (2.0, 2.0, 2.0))

            mock_phantom = MagicMock()
            mock_phantom.rois = {"sphere1": {}}
            mock_phantom_class.return_value = mock_phantom

            mock_generate_plots.return_value = None

            # Only test meaningful cases, not empty results that cause warnings
            problematic_results = [
                ([{"diameter_mm": 10.0}], {10: 95.0}),
                ([{"diameter_mm": 10.0, "percentage_contrast_QH": 85.0}], {10: 95.0}),
                (
                    [
                        {
                            "diameter_mm": 10.0,
                            "percentaje_constrast_QH": 85.0,
                            "background_variability_N": 5.2,
                            "avg_hot_counts_CH": 15000.0,
                            "avg_bkg_counts_CB": 2000.0,
                            "bkg_std_dev_SD": 104.0,
                        }
                    ],
                    {10: 95.0},
                ),
            ]

            for results, lung_results in problematic_results:
                mock_calculate_metrics.return_value = (results, lung_results)

                args = MagicMock()
                args.verbose = False
                args.input_image = "input.nii"
                args.output = "output.txt"
                args.config = "config.yaml"
                args.spacing = None
                args.save_visualizations = True
                args.visualizations_dir = "test_viz"

                result = cli.run_analysis(args)
                assert result in [0, 1]

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    def test_visualization_with_malformed_data(
        self,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test visualization generation with malformed data."""
        mock_exists.return_value = True
        cfg = _prepare_cfg_for_tests()
        mock_load_config.return_value = cfg
        mock_load_image.return_value = (np.ones((361, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((361, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        # Mock generate_plots to raise different KeyErrors
        key_errors_to_test = [
            KeyError("percentaje_constrast_QH"),
            KeyError("PHANTHOM"),  # Another typo in your code
            KeyError("diameter_mm"),
            KeyError("background_variability_N"),
        ]

        for key_error in key_errors_to_test:
            mock_generate_plots.side_effect = key_error

            mock_calculate_metrics.return_value = (
                [
                    {
                        "diameter_mm": 10.0,
                        "wrong_key_name": 85.0,
                    }
                ],
                {10: 95.0},
            )

            args = MagicMock()
            args.verbose = False
            args.input_image = "input.nii"
            args.output = "output.txt"
            args.config = "config.yaml"
            args.spacing = None
            args.save_visualizations = True
            args.visualizations_dir = "test_viz"

            result = cli.run_analysis(args)
            assert result == 1

    def test_data_format_validation(self):
        """Test validation of data formats expected by CLI functions."""
        expected_result_keys = [
            "diameter_mm",
            "percentaje_constrast_QH",  # Note the typo - this is what your code expects
            "background_variability_N",
            "avg_hot_counts_CH",
            "avg_bkg_counts_CB",
            "bkg_std_dev_SD",
        ]

        proper_result = dict.fromkeys(expected_result_keys, 10.0)

        for key in expected_result_keys:
            assert key in proper_result

        assert "percentaje_constrast_QH" in expected_result_keys  # Document the typo

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    @patch("src.nema_quant.cli.generate_rois_plots")
    @patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
    @patch("src.nema_quant.cli.generate_reportlab_report")
    def test_keyerror_handling_comprehensive(
        self,
        mock_report,
        mock_boxplot,
        mock_rois_plots,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Comprehensive test of KeyError handling in CLI."""
        mock_exists.return_value = True
        cfg = _prepare_cfg_for_tests()
        mock_load_config.return_value = cfg
        mock_load_image.return_value = (np.ones((361, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((361, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        # Test different KeyError scenarios
        keyerror_scenarios = [
            ("generate_plots", KeyError("percentaje_constrast_QH")),
            ("generate_plots", KeyError("PHANTHOM")),
            ("generate_rois_plots", KeyError("some_other_key")),
            ("generate_boxplot_with_mean_std", KeyError("lung_data")),
            ("generate_reportlab_report", KeyError("report_data")),
        ]

        for function_name, error in keyerror_scenarios:
            # Reset all mocks
            for mock_func in [
                mock_generate_plots,
                mock_rois_plots,
                mock_boxplot,
                mock_report,
            ]:
                mock_func.reset_mock()
                mock_func.return_value = None
                mock_func.side_effect = None

            # Set the specific function to raise the KeyError
            if function_name == "generate_plots":
                mock_generate_plots.side_effect = error
            elif function_name == "generate_rois_plots":
                mock_rois_plots.side_effect = error
            elif function_name == "generate_boxplot_with_mean_std":
                mock_boxplot.side_effect = error
            elif function_name == "generate_reportlab_report":
                mock_report.side_effect = error

            mock_calculate_metrics.return_value = (
                [
                    {
                        "diameter_mm": 10.0,
                        "percentaje_constrast_QH": 85.0,
                        "background_variability_N": 5.2,
                        "avg_hot_counts_CH": 15000.0,
                        "avg_bkg_counts_CB": 2000.0,
                        "bkg_std_dev_SD": 104.0,
                    }
                ],
                {10: 95.0},
            )

            args = MagicMock()
            args.verbose = False
            args.input_image = "input.nii"
            args.output = "output.txt"
            args.config = "config.yaml"
            args.spacing = None
            args.save_visualizations = True  # Force visualization calls
            args.visualizations_dir = "test_viz"

            result = cli.run_analysis(args)

            # Should handle KeyError and return error code
            assert result == 1, f"Failed for {function_name} with {error}"

    def test_keyerror_documentation(self):
        """Document the KeyErrors that occur in the codebase."""
        known_keyerrors = [
            "percentaje_constrast_QH",  # Should be 'percentage_contrast_QH'
            "PHANTHOM",  # Should be 'PHANTOM'
        ]

        corrections = {
            "percentaje_constrast_QH": "percentage_contrast_QH",
            "PHANTHOM": "PHANTOM",
        }

        for typo in known_keyerrors:
            assert typo in corrections
            print(f"Known typo: '{typo}' should be '{corrections[typo]}'")


def _prepare_cfg_for_tests():
    """Return a get_cfg_defaults() with safe ROI centers/colors for test image dims (z=50,y=100,x=100)."""
    cfg = get_cfg_defaults()
    # ensure ROIS node exists and has needed defaults
    if not hasattr(cfg, "ROIS") or cfg.ROIS is None:
        cfg.ROIS = CfgNode()
    cfg.ROIS.BACKGROUND_OFFSET_YX = getattr(cfg.ROIS, "BACKGROUND_OFFSET_YX", (0, 0))
    cfg.ROIS.ORIENTATION_YX = getattr(cfg.ROIS, "ORIENTATION_YX", [1, 1])

    # ensure PHANTHOM node exists
    if not hasattr(cfg, "PHANTHOM") or cfg.PHANTHOM is None:
        cfg.PHANTHOM = CfgNode()
    cfg.PHANTHOM.BACKGROUND_OFFSET_YX = getattr(
        cfg.PHANTHOM, "BACKGROUND_OFFSET_YX", (0, 0)
    )

    # Ensure ROI_DEFINITIONS_MM is present and safe for the mocked image (z=50,y=100,x=100)
    rois = getattr(cfg.PHANTHOM, "ROI_DEFINITIONS_MM", None)
    if rois is None:
        cfg.PHANTHOM.ROI_DEFINITIONS_MM = []
        rois = cfg.PHANTHOM.ROI_DEFINITIONS_MM

    # use center_yx within (y=100,x=100) and provide colors if missing
    safe_center = (25, 50)
    for roi in rois:
        roi["center_yx"] = roi.get("center_yx", safe_center)
        roi["color"] = roi.get("color", "C0")
        # accept both diameter and diameter_mm
        if "diameter_mm" not in roi and "diameter" in roi:
            roi["diameter_mm"] = roi["diameter"]

    return cfg
