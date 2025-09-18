import inspect

from src.nema_quant import cli, reporting


def test_reporting_function_signatures():
    """Document actual function signatures for debugging."""
    functions = [
        "save_results_to_txt",
        "generate_reportlab_report",
        "generate_plots",
        "generate_rois_plots",
        "generate_boxplot_with_mean_std",
    ]

    signatures = {}
    for func_name in functions:
        if hasattr(reporting, func_name):
            func = getattr(reporting, func_name)
            try:
                sig = inspect.signature(func)
                signatures[func_name] = {
                    "signature": str(sig),
                    "parameters": list(sig.parameters.keys()),
                }
            except Exception as e:
                signatures[func_name] = f"Error: {e}"  # type: ignore

    print(f"\nReporting function signatures: {signatures}")
    assert len(signatures) > 0


def test_cli_function_signatures():
    """Document CLI function signatures for debugging."""
    functions = [
        "run_analysis",
        "main",
        "create_parser",
        "setup_logging",
        "load_configuration",
    ]

    signatures = {}
    for func_name in functions:
        if hasattr(cli, func_name):
            func = getattr(cli, func_name)
            try:
                sig = inspect.signature(func)
                signatures[func_name] = {
                    "signature": str(sig),
                    "parameters": list(sig.parameters.keys()),
                }
            except Exception as e:
                signatures[func_name] = f"Error: {e}"  # type: ignore

    print(f"\nCLI function signatures: {signatures}")
    assert len(signatures) > 0
