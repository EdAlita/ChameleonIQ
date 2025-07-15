import numpy as np
from typing import Tuple
from scipy.ndimage import center_of_mass, binary_fill_holes, label


def find_phantom_center(
    image_data_3d: np.ndarray,
    threshold_ratio: float = 0.2
) -> Tuple[float, float, float]:
    """
    Encuentra el centro del fantoma usando un enfoque morfológico robusto.

    Este método mejorado utiliza el relleno de
    agujeros para crear una máscara sólida
    del cuerpo del fantoma, ignorando las
    estructuras internas (esferas calientes)
    que pueden sesgar el cálculo del centro de masa.

    Parameters
    ----------
    image_data_3d : np.ndarray
        Los datos de la imagen en un array 3D de NumPy (z, y, x).
    threshold_ratio : float, optional
        Factor para determinar el umbral. Por defecto es 0.5.

    Returns
    -------
    tuple of float
        Las coordenadas del centroide (z, y, x) del fantoma.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 09:32:08
    """
    if image_data_3d.ndim != 3:
        raise ValueError("La imagen de entrada debe ser un array 3D (z,y,x).")

    max_val = np.max(image_data_3d)
    binary_mask = image_data_3d > (max_val * threshold_ratio)
    filled_mask = binary_fill_holes(binary_mask)
    labeled_mask, num_features = label(filled_mask)  # type: ignore[misc]

    if num_features == 0:
        raise RuntimeError(
            "No se pudo encontrar ningún objeto"
            "en la imagen con el umbral actual."
        )

    largest_label = np.argmax(np.bincount(labeled_mask.ravel())[1:]) + 1
    phantom_mask = labeled_mask == largest_label

    centroid_zyx_raw = tuple(float(x) for x in center_of_mass(phantom_mask))

    if not isinstance(centroid_zyx_raw, tuple) or len(centroid_zyx_raw) != 3:
        raise RuntimeError("El centroide calculado no tiene 3 dimensiones.")

    centroid_zyx: Tuple[float, float, float] = (
        float(centroid_zyx_raw[0]),
        float(centroid_zyx_raw[1]),
        float(centroid_zyx_raw[2])
    )

    return centroid_zyx


def voxel_to_mm(
    voxel_indices_zyx: Tuple[int, int, int],
    image_dims_xyz: Tuple[int, int, int],
    voxel_spacing_xyz: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Convierte índices de vóxel (orden z,y,x) a
    coordenadas físicas (mm, relativas al centro).
    Esta función se considera la "verdad fundamental" para la conversión.
    Se asume que la coordenada del vóxel representa el centro de ese vóxel.

    Parameters
    ----------
    voxel_indices_zyx : tuple of int
        Los índices del vóxel en orden (z, y, x).
    image_dims_xyz : tuple of int
        Las dimensiones totales de la imagen en vóxeles (dim_x, dim_y, dim_z).
    voxel_spacing_xyz : tuple of float
        El tamaño del vóxel en mm (spacing_x, spacing_y, spacing_z).

    Returns
    -------
    tuple of float
        Las coordenadas físicas (x, y, z) en milímetros desde el centro.
    """
    # El centro del volumen de la imagen en índices de vóxel (puede ser .5)
    center_vox_x = (image_dims_xyz[0] - 1) / 2.0
    center_vox_y = (image_dims_xyz[1] - 1) / 2.0
    center_vox_z = (image_dims_xyz[2] - 1) / 2.0

    # Desplazamiento del vóxel dado desde el centro, en unidades de vóxel
    offset_vox_x = voxel_indices_zyx[2] - center_vox_x
    offset_vox_y = voxel_indices_zyx[1] - center_vox_y
    offset_vox_z = voxel_indices_zyx[0] - center_vox_z

    # Convertir el desplazamiento en vóxeles a un desplazamiento en mm
    mm_x = offset_vox_x * voxel_spacing_xyz[0]
    mm_y = offset_vox_y * voxel_spacing_xyz[1]
    mm_z = offset_vox_z * voxel_spacing_xyz[2]

    return (mm_x, mm_y, mm_z)


def mm_to_voxel(
    mm_coords: Tuple[float, float, float],
    image_dims_xyz: Tuple[int, int, int],
    voxel_spacing_xyz: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """
    Convierte coordenadas físicas (en mm, relativas al centro)
    a los índices del vóxel más cercano.
    Esta es la función inversa de voxel_to_mm.

    Parameters
    ----------
    mm_coords : tuple of float
        Las coordenadas (x, y, z) en milímetros desde el centro.
    image_dims_xyz : tuple of int
        Las dimensiones totales de la imagen en vóxeles (dim_x, dim_y, dim_z).
    voxel_spacing_xyz : tuple of float
        El tamaño del vóxel en mm (spacing_x, spacing_y, spacing_z).

    Returns
    -------
    tuple of int
        Los índices del vóxel correspondientes en orden (z, y, x).

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 09:13:50
    """
    # El centro del volumen de la imagen en índices de vóxel (puede ser .5)
    center_vox_x = (image_dims_xyz[0] - 1) / 2.0
    center_vox_y = (image_dims_xyz[1] - 1) / 2.0
    center_vox_z = (image_dims_xyz[2] - 1) / 2.0

    # Convertir el desplazamiento en mm a un desplazamiento en vóxeles
    offset_vox_x = mm_coords[0] / voxel_spacing_xyz[0]
    offset_vox_y = mm_coords[1] / voxel_spacing_xyz[1]
    offset_vox_z = mm_coords[2] / voxel_spacing_xyz[2]

    # Calcular el índice final del vóxel y redondear al entero más cercano
    # np.round maneja correctamente los casos .5
    final_vox_x = int(np.round(center_vox_x + offset_vox_x))
    final_vox_y = int(np.round(center_vox_y + offset_vox_y))
    final_vox_z = int(np.round(center_vox_z + offset_vox_z))

    # Devuelve en el orden (z, y, x) para la indexación de NumPy
    return (final_vox_z, final_vox_y, final_vox_x)


if __name__ == '__main__':
    # --- Ejemplo de Uso y Verificación con los valores del usuario ---
    dims = (392, 392, 346)
    spacing = (2.0644, 2.0644, 2.0644)

    center_voxel_indices = mm_to_voxel((0.0, 0.0, 0.0), dims, spacing)

    print(f"Dimensiones de la imagen (x,y,z): {dims}")
    print(f"Espaciado de vóxel (x,y,z): {spacing}")
    print("-" * 30)

    # --- Prueba de conversión: mm -> vóxel ---
    print(f"El vóxel más cercano al centro físico (0,0,0) es:"
          f" {center_voxel_indices} (z,y,x)")

    # --- Prueba de conversión inversa: vóxel -> mm ---
    print(f"\nConvirtiendo los índices del vóxel central"
          f"{center_voxel_indices} de vuelta a mm...")
    calculated_mm = voxel_to_mm(center_voxel_indices, dims, spacing)
    print(
        f"  -> Coordenada calculada: ({calculated_mm[0]:.2f}, "
        f"{calculated_mm[1]:.2f}, {calculated_mm[2]:.2f}) (x,y,z)"
    )

    # --- Verificación de simetría ---
    print("\nVerificando que las conversiones"
          " son inversas la una de la otra...")
    test_mm = (66.08, 34.56, -2.90)
    print(f"Test 1: {test_mm} -> Vóxel -> mm")
    voxel_result = mm_to_voxel(test_mm, dims, spacing)
    mm_result = voxel_to_mm(voxel_result, dims, spacing)
    print(
        f"  {test_mm} -> {voxel_result} -> "
        f"({mm_result[0]:.2f}, {mm_result[1]:.2f}, {mm_result[2]:.2f})"
    )
    assert np.allclose(test_mm, mm_result, atol=1.1), "Conversión inv. falló!"
    print("  -> Verificación exitosa!")

    test_voxel = (169, 211, 171)
    print(f"Test 2: {test_voxel} -> mm -> Vóxel")
    mm_result_2 = voxel_to_mm(test_voxel, dims, spacing)
    voxel_result_2 = mm_to_voxel(mm_result_2, dims, spacing)
    print(f"  {test_voxel} -> {mm_result_2} -> {voxel_result_2}")
    assert test_voxel == voxel_result_2, "Conversión inv. falló!"
    print("  -> Verificación exitosa!")
