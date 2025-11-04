#!/usr/bin/env python
"""
:Summary:
:Description:
In all of the code below, A is seen as ground truth/gold standard for comparison
Selected the ones that work in extreme cases (no segmentation, i.e. ground truth vs empty image)
This is to avoid true negative issue in 'large' images, where the majority should have the label 0
filelist: man outline, aut outline
:Requires:
Update this meassage
Code base on EISRAD implementation
__author__ = 'mds'
__contact__ = 'software@markus-schirmer.com'
__copyright__ = ''
__license__ = ''
__date__ = '2019-06'
__version__ = '1.0'
#===========================
"""
from collections import OrderedDict

import numpy as np
import skimage.measure as skm
from scipy.ndimage import _ni_support  # type: ignore[attr-defined]
from scipy.ndimage.morphology import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


def get_truepos(A, B):
    """Return true positive"""
    return np.float64(np.sum(np.logical_and((B == 1), (A == 1)).astype(int)))


def get_trueneg(A, B):
    """Return true negative"""
    return np.float64(np.sum(np.logical_and((B == 0), (A == 0)).astype(int)))


def get_falsepos(A, B):
    """Return false positive"""
    return np.float64(np.sum(np.logical_and((B == 1), (A == 0)).astype(int)))


def get_falseneg(A, B):
    """Return false negative"""
    return np.float64(np.sum(np.logical_and((A == 1), (B == 0)).astype(int)))


def get_dice(A, B):
    """Return Dice coefficient"""
    TP = get_truepos(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    denominator = 2.0 * TP + FP + FN
    numerator = 2.0 * TP
    if denominator > 0:
        return numerator / denominator
    else:
        return np.nan


def get_jaccard(A, B):
    """Return Jaccard index"""
    TP = get_truepos(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    denominator = TP + FP + FN
    numerator = TP

    if denominator > 0:
        return numerator / denominator
    else:
        return np.nan


def get_sensitivity(A, B):
    """Return sensitivity"""
    TP = get_truepos(A, B)
    FN = get_falseneg(A, B)

    return TP / (TP + FN)


def get_F1(A, B):
    """Return F1 score"""
    TP = get_truepos(A, B)
    FP = get_falseneg(A, B)
    FN = get_falsepos(A, B)
    return 2 * TP / (2 * TP + FP + FN)


def get_specificity(A, B):
    """Return specificity"""
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)

    return TN / (TN + FP)


def get_global_consistency_error(A, B):
    """Return global consistency error"""
    n = float(A.size)

    TP = get_truepos(A, B)
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    E1 = (FN * (FN + 2 * TP) / (TP + FN) + (FP * (FP + 2 * TN)) / (TN + FP)) / n
    E2 = (FP * (FP + 2 * TP) / (TP + FP) + FN * (FN + 2 * TN) / (TN + FN)) / n

    return np.min([E1, E2])


def get_volumetric_similarity(A, B):
    """Return volumetric similarity"""
    TP = get_truepos(A, B)
    # _TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    return 1.0 - np.abs(FN - FP) / (2.0 * TP + FP + FN)


def get_abcd(A, B):
    n = float(A.size)
    TP = get_truepos(A, B)
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    a = 0.5 * (TP * (TP - 1) + FP * (FP - 1) + TN * (TN - 1) + FN * (FN - 1))
    b = 0.5 * ((TP + FN) ** 2 + (TN + FP) ** 2 - (TP**2 + TN**2 + FP**2 + FN**2))
    c = 0.5 * ((TP + FP) ** 2 + (TN + FN) ** 2 - (TP**2 + TN**2 + FP**2 + FN**2))
    d = n * (n - 1.0) / 2.0 - (a + b + c)

    return a, b, c, d


def get_rand_idx(A, B):
    # get a, b, c and d
    a, b, c, d = get_abcd(A, B)

    RI = (a + b) / (a + b + c + d)
    ARI = 2 * (a * d - b * c) / (c**2 + b**2 + 2 * a * d + (a + d) * (c + b))

    return RI, ARI


def get_probabilities(A, B):
    n = float(A.size)
    TP = get_truepos(A, B)
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    # p: S_g^1, S_g^2, S_t^1, S_t^2)
    p = []
    p.append((TP + FN) / n)
    p.append((TN + FN) / n)
    p.append((TP + FP) / n)
    p.append((TN + FP) / n)

    # p: (S_g^1, S_t^1), (S_g_1 S_t^2), (S_t^2, S_g^1) , (S_g^1,S_t^2)
    p.append(TP / n)
    p.append(FN / n)
    p.append(FP / n)
    p.append(TN / n)

    return p


def get_log(p):
    """Return base-2 logarithm multiplied by exponent.
    If exponent is 0, then return 0
    """
    if p == 0:
        return 0.0
    return p * np.log2(p)


def get_MI(A, B):
    """Return mutual information"""
    # get probabilities
    p = [get_log(ii) for ii in get_probabilities(A, B)]

    H_1 = -(p[0] + p[1])
    H_2 = -(p[2] + p[3])
    H_12 = -(np.sum(p[4:]))

    MI = H_1 + H_2 - H_12
    VOI = (H_1 + H_2 - 2 * MI) / (2 * np.log2(2.0))

    return 2 * MI / (H_1 + H_2), VOI


def get_ICC(A, B):
    """Return intraclass correlation coefficient"""
    n = float(A.size)
    mean_img = (A + B) / 2.0

    MS_w = np.sum((A - mean_img) ** 2 + (B - mean_img) ** 2) / n
    MS_b = 2 / (n - 1) * np.sum((mean_img - np.mean(mean_img)) ** 2)

    numerator = MS_b - MS_w
    denominator = MS_b + MS_w

    if denominator > 0:
        return numerator / denominator
    else:
        return np.nan


def get_PBD(A, B):
    """Return probabilistic distance"""
    combined = np.sum(np.multiply(A, B))
    if combined == 0:
        return 1

    return np.sum(np.abs(A - B)) / (2.0 * combined)


def get_KAP(A, B):
    """Return Cohen's kappa"""
    n = float(A.size)
    TP = get_truepos(A, B)
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    fa = TP + TN
    fc = 1.0 / n * ((TN + FN) * (TN + FP) + (FP + TP) * (FN + TP))

    return (fa - fc) / (n - fc)


def get_AUC(A, B):
    """Return AUC"""
    # n = float(A.size)
    TP = get_truepos(A, B)
    TN = get_trueneg(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)

    return 1.0 - 0.5 * (FP / (FP + TN) + FN / (FN + TP))


def directed_HD(A, B):
    """Calculate the Hausdorff distance"""
    # get coordinates
    coords_A = np.vstack(np.where(A)).transpose()
    coords_B = np.vstack(np.where(B)).transpose()
    if (len(coords_A) == 0) and (len(coords_B) == 0):
        return 1.0
    if (len(coords_A) == 0) or (len(coords_B) == 0):
        return 1.0

    # normalize by max possible distance
    # max_distance = float(np.sqrt(np.sum(np.asarray(A.shape) ** 2)))

    # calculate all distances between points in A and B
    min_dist = []
    for ii in np.arange(coords_A.shape[0]):
        min_dist.append(
            np.min(np.sqrt(np.sum((coords_B - coords_A[ii, :]) ** 2, axis=1)))
        )

    return min_dist


def get_HD_v1(A, B):
    """Return the Hausdorff distance"""
    HD_AB = np.max(directed_HD(A, B))
    HD_BA = np.max(directed_HD(B, A))

    return np.max([HD_AB, HD_BA])


def get_AVD(A, B):
    HD_AB = np.mean(directed_HD(A, B))
    HD_BA = np.mean(directed_HD(B, A))

    return np.max([HD_AB, HD_BA])


def get_ODER(A, B):
    """Return the detection and outline error rates
    see Wack et al. 2012 - Improved assessment of multiple sclerosis lesion segmentation agreement via detection and outline error estimates
    """

    # mean area of raters
    MTA = (np.sum(A) + np.sum(B)) / 2.0

    # intersection of outlines
    intersect = np.multiply(A, B)

    # regions in A
    labels_A = skm.label(A)

    # regions in B
    labels_B = skm.label(B)

    # labels in found in A but also in B
    labels_in_A_and_B = np.unique(np.multiply(intersect, labels_A))
    labels_in_B_and_A = np.unique(np.multiply(intersect, labels_B))

    # labels unique in A and unique in B
    labels_only_in_A = np.asarray(
        [ii for ii in np.unique(labels_A) if ii not in labels_in_A_and_B]
    )
    labels_only_in_B = np.asarray(
        [ii for ii in np.unique(labels_B) if ii not in labels_in_B_and_A]
    )

    # make sure 0 is not picked up
    labels_in_A_and_B = labels_in_A_and_B[labels_in_A_and_B > 0]
    labels_in_B_and_A = labels_in_B_and_A[labels_in_B_and_A > 0]
    labels_only_in_A = labels_only_in_A[labels_only_in_A > 0]
    labels_only_in_B = labels_only_in_B[labels_only_in_B > 0]

    # calculate detection error
    # sum of areas only picked up by A plus sum of areas only picked up by B
    DE = np.sum([np.sum(labels_A == ii) for ii in labels_only_in_A]) + np.sum(
        [np.sum(labels_B == ii) for ii in labels_only_in_B]
    )

    # calculate outline error
    # total difference between union and intersection of the region that was outlines by both
    # = area determined by rater 1 + area determined by rater b - 2 * area determined by both
    # as union is area determined by rater 1 + area determined by rater b - area determined by both
    OE = (
        np.sum([np.sum(labels_A == ii) for ii in labels_in_A_and_B])
        + np.sum([np.sum(labels_B == ii) for ii in labels_in_B_and_A])
        - 2 * np.sum(intersect)
    )

    # convert to rates and return
    return OE / MTA, DE / MTA


def calculate_metrics(label_arr, sub_arr, metrics, num_classes):

    labels = np.arange(num_classes)
    label_metrics = np.zeros((1, len(labels) * len(metrics)))
    print(label_metrics.size)
    colums_names = []
    for idx, label in enumerate(labels):
        if label == 0:
            background_flag = True
        else:
            background_flag = False
        label_map = np.zeros(label_arr.shape)
        label_map[label_arr == label] = 1
        pred_map = np.zeros(label_arr.shape)
        pred_map[sub_arr == label] = 1
        values = get_values(label_map, pred_map, metrics, background_flag)
        for idj, key in enumerate(values.keys()):
            print(idx * len(metrics) + idj)
            label_metrics[0, (idx * len(metrics)) + idj] = values[key]
            colums_names.append(str(key) + "_" + str(int(label)))

    return label_metrics, colums_names


def get_values(A, B, measures, background=False, voxelspacing=None):
    """Return all similarity metric values binary maps"""
    # initialise
    values = OrderedDict()
    # run through list of implementations
    if "Dice" in measures:
        values["Dice"] = get_dice(A, B)
    if "Jaccard" in measures:
        values["Jaccard"] = get_jaccard(A, B)
    if "1-GCE" in measures:
        values["1-GCE"] = 1.0 - get_global_consistency_error(A, B)
    if "VS" in measures:
        values["VS"] = get_volumetric_similarity(A, B)

    if ("1-VOI" in measures) or ("MI" in measures):
        NMI, VOI = get_MI(A, B)
        if "1-VOI" in measures:
            values["1-VOI"] = 1.0 - VOI
        if "MI" in measures:
            values["MI"] = NMI
    if "ICC" in measures:
        values["ICC"] = get_ICC(A, B)
    if "KAP" in measures:
        values["KAP"] = get_KAP(A, B)
    if "AUC" in measures:
        values["AUC"] = get_AUC(A, B)
    if ("RI" in measures) or ("ARI" in measures):
        NRI, ARI = get_rand_idx(A, B)
        if "RI" in measures:
            values["RI"] = NRI
        if "ARI" in measures:
            values["ARI"] = ARI
    if "1/(1+AVD)" in measures:
        if not background:
            values["1/(1+AVD)"] = 1.0 / (1.0 + get_AVD(A, B))
        else:
            values["1/(1+AVD)"] = np.nan
    if "AVD" in measures:
        if not background:
            values["AVD"] = get_AVD(A, B)
        else:
            values["AVD"] = np.nan
    if "HD" in measures or "HD95" in measures:
        hd = directed_HD(A, B)
        _, h95 = get_HD(A, B, voxelspacing=voxelspacing)
        if "HD" in measures and not background:
            values["HD"] = hd
        else:
            values["HD"] = np.nan

        if "HD95" in measures and not background:
            values["HD95"] = h95
        else:
            values["HD95"] = np.nan
    if "F1" in measures:
        if not background:
            values["F1"] = get_F1(A, B)
        else:
            values["F1"] = np.nan
    if "Recall" in measures:
        if not background:
            values["Recall"] = get_sensitivity(A, B)
        else:
            values["Recall"] = np.nan
    if "ASSD" in measures:
        if not background:
            values["ASSD"] = assd(A, B, voxelspacing=voxelspacing)
        else:
            values["ASSD"] = np.nan
    return values


# https://github.com/amanbasu/3d-prostate-segmentation/blob/master/metric_eval.py
def get_HD(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`assd`
    :func:`asd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    # test for emptiness
    if 0 == np.count_nonzero(result) and 0 == np.count_nonzero(reference):
        return 0, 0
    if 0 == np.count_nonzero(reference) and np.count_nonzero(result) > 0:
        return np.nan, np.nan
    if np.count_nonzero(reference) > 0 and np.count_nonzero(result) == 0:
        return np.nan, np.nan

    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd = max(hd1.max(), hd2.max())
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    # hd50 = np.percentile(np.hstack((hd1, hd2)), 50)
    return hd, hd95


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    # test for emptiness
    if 0 == np.count_nonzero(result) and 0 == np.count_nonzero(reference):
        return 0
    if 0 == np.count_nonzero(reference) and np.count_nonzero(result) > 0:
        return np.nan
    if np.count_nonzero(reference) > 0 and np.count_nonzero(result) == 0:
        return np.nan

    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    # binary_mask = binary_dilation(reference, structure=generate_binary_structure(rank=3, connectivity=1),iterations=2).astype(result.dtype)
    # result = np.logical_and(result, binary_mask)
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]  # type: ignore[index]

    return sds


def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`asd`
    :func:`hd`

    Notes
    -----
    This is a real metric, obtained by calling and averaging

    >>> asd(result, reference)

    and

    >>> asd(reference, result)

    The binary images can therefore be supplied in any order.
    """
    assd = np.mean(
        (
            asd(result, reference, voxelspacing, connectivity),
            asd(reference, result, voxelspacing, connectivity),
        )
    )
    return assd


def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`hd`


    Notes
    -----
    This is not a real metric, as it is directed. See `assd` for a real metric of this.

    The method is implemented making use of distance images and simple binary morphology
    to achieve high computational speed.

    Examples
    --------
    The `connectivity` determines what pixels/voxels are considered the surface of a
    binary object. Take the following binary image showing a cross

    >>> from scipy.ndimage.morphology import generate_binary_structure
    >>> cross = generate_binary_structure(2, 1)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

    With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    object surface, resulting in the surface

    .. code-block:: python

        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])

    Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:

    .. code-block:: python

        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])

    , as a diagonal connection does no longer qualifies as valid object surface.

    This influences the  results `asd` returns. Imagine we want to compute the surface
    distance of our cross to a cube-like object:

    >>> cube = generate_binary_structure(2, 1)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])

    , which surface is, independent of the `connectivity` value set, always

    .. code-block:: python

        array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])

    Using a `connectivity` of `1` we get

    >>> asd(cross, cube, connectivity=1)
    0.0

    while a value of `2` returns us

    >>> asd(cross, cube, connectivity=2)
    0.20000000000000001

    due to the center of the cross being considered surface as well.

    """
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd
