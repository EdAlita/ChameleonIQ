Batch Processing
================

Process Multiple Images
-----------------------

Command Line
~~~~~~~~~~~~

Analyze all images in a directory::

    nema_quant --input-dir images/ --output-dir results/ --parallel

With options::

    nema_quant --input-dir images/ \
               --output-dir results/ \
               --config config.yaml \
               --parallel \
               --num-workers 4

Python API
~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from nema_quant import analysis
    from concurrent.futures import ProcessPoolExecutor

    image_dir = Path('images/')
    output_dir = Path('results/')
    output_dir.mkdir(exist_ok=True)

    def process_image(image_path):
        phantom = NemaPhantom(image_path=str(image_path))
        metrics = analysis.calculate_image_quality_metrics(phantom)
        return metrics

    # Process in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        images = list(image_dir.glob('*.nii.gz'))
        results = executor.map(process_image, images)

    # Save results
    for img, result in zip(images, results):
        output_file = output_dir / f"{img.stem}_metrics.json"
        result.save(output_file)
