from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Flowable
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from typing import List, Dict, Any


def generate_pdf_report(
    output_path: str,
    analysis_results: List[Dict[str, Any]],
    acquisition_params: Dict[str, Any],
    reconstruction_params: Dict[str, Any],
    background_concentration: float,
    manufacturer_dose: str,
    sphere_images: List[str],
):
    """
    Generates a PDF report for NEMA quality analysis results.

    Parameters
    ----------
    output_path : str
        Path to save the generated PDF.
    analysis_results : list of dict
        Results of NEMA analysis for each sphere, including metrics
        like percent contrast and background variability.
    acquisition_params : dict
        Acquisition parameters including emission
        imaging time, axial step size, etc.
    reconstruction_params : dict
        Reconstruction parameters including filters applied, pixel size, etc.
    background_concentration : float
        Background concentration used to fill the phantom.
    manufacturer_dose : str
        Manufacturer's recommended injected dose for total body studies.
    sphere_images : list
        Paths to images (transverse and coronal) for the spheres.
    """
    # Create the document template
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements: list[Flowable] = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    header_style = styles["Heading2"]
    body_style = styles["BodyText"]

    # Title
    title = Paragraph("NEMA Quality Analysis Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Section: Background concentration and dose
    background_text = (
        "<b>Background Concentration and Dose</b><br/>"
        f"Background Concentration: {background_concentration} kBq/ml<br/>"
        f"Manufacturer's Recommended Dose: {manufacturer_dose}"
    )
    elements.append(Paragraph(background_text, body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Section: Acquisition Parameters
    acquisition_text = "<b>Acquisition Parameters</b><br/>"
    for param, value in acquisition_params.items():
        acquisition_text += f"{param}: {value}<br/>"
    elements.append(Paragraph(acquisition_text, body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Section: Reconstruction Parameters
    reconstruction_text = "<b>Reconstruction Parameters</b><br/>"
    for param, value in reconstruction_params.items():
        reconstruction_text += f"{param}: {value}<br/>"
    elements.append(Paragraph(reconstruction_text, body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Section: Analysis Results Table
    elements.append(Paragraph("<b>Analysis Results</b>", header_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Create the table data
    table_data = [
        [
            "Sphere Size\n(mm)", "Percent Contrast\n(QH)",
            "Background Variability\n(N)", "Average Hot Counts\n(CH)",
            "Average Background Counts\n(CB)"
        ]
    ]

    for result in analysis_results:
        table_data.append([
            result.get("diameter_mm", "Unknown"),
            f"{result.get('percentaje_constrast_QH', 'N/A'):.2f}%",
            f"{result.get('background_variability_N', 'N/A'):.2f}%",
            f"{result.get('avg_hot_counts_CH', 'N/A'):.2f}",
            f"{result.get('avg_bkg_counts_CB', 'N/A'):.2f}",
        ])

    # Create the table
    column_widths = [1.5 * inch] * 5
    table = Table(table_data, colWidths=column_widths)

    # Style the table
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.beige, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))
    """
    # Section: Sphere Images
    elements.append(Paragraph("<b>Sphere Images</b>", header_style))
    for img_path in sphere_images:
        from reportlab.platypus import Image
        elements.append(Image(img_path, width=4 * inch, height=4 * inch))
        elements.append(Spacer(1, 0.3 * inch))
    """
    # Build the document
    doc.build(elements)


if __name__ == "__main__":
    # Example data
    analysis_results = [
        {
            "diameter_mm": 10,
            "percentaje_constrast_QH": 95.0,
            "background_variability_N": 5.0,
            "avg_hot_counts_CH": 800.0,
            "avg_bkg_counts_CB": 100.0,
        },
        {
            "diameter_mm": 17,
            "percentaje_constrast_QH": 90.0,
            "background_variability_N": 4.5,
            "avg_hot_counts_CH": 750.0,
            "avg_bkg_counts_CB": 105.0,
        },
    ]
    acquisition_params = {
        "Emission Imaging Time": "10 minutes",
        "Axial Step Size": "5 mm",
        "Total Axial Imaging Distance": "200 mm",
    }
    reconstruction_params = {
        "Filters Applied": "Gaussian smoothing",
        "Pixel Size": "2 mm",
        "Image Matrix Size": "128 x 128",
        "Slice Thickness": "2 mm",
    }
    background_concentration = 2.5
    manufacturer_dose = "370 MBq"
    sphere_images = ["path_to_transverse_image.png",
                     "path_to_coronal_image.png"]

    # Generate PDF
    generate_pdf_report(
        "nema_quality_analysis_report.pdf",
        analysis_results,
        acquisition_params,
        reconstruction_params,
        background_concentration,
        manufacturer_dose,
        sphere_images,
    )
