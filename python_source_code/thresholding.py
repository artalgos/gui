import numpy as np
from scipy import ndimage


def RosinThreshold(imhist, picknonempty=0):
    mmax2 = np.amax(imhist)
    mpos = imhist.argmax()
    p1 = [mpos, mmax2]
    L = imhist.shape[0]
    lastbin = mpos
    for i in np.arange(start=mpos, stop = L):
        if(imhist[i] > 0):
            lastbin = i
    p2 = [lastbin, imhist[lastbin]]
    DD = np.sqrt(np.array(p2[0] - p1[0], dtype='int64')**2 + \
             np.array(p2[1] - p1[1], dtype='int64')**2)
    if DD != 0:
        best = -1
        found = -1
        for i in np.arange(start=mpos, stop=lastbin+1):
            p0 = [i, imhist[i]]
            d = np.abs((p2[0] - p1[0])*(p1[1]-p0[1]) - (p1[0]-p0[0])*(p2[1]-p1[1]))
            d = d/DD
            if ((d > best) and ((imhist[i]>0) or (picknonempty == 0))):
                best = d
                found = i
        if found == -1:
            found = lastbin+1
    else:
        found = lastbin+1
    T = np.min([found+1, L])
    return(T)

def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to `image`.
    This algorithm finds regions where `image` is greater than `high`
    OR `image` is greater than `low` *and* that region is connected to
    a region greater than `high`.
    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float
        Lower threshold.
    high : float
        Higher threshold.
    Returns
    -------
    thresholded : array of bool, same shape as `image`
        Array in which `True` indicates the locations where `image`
        was above the hysteresis threshold.
    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])
    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           DOI: 10.1109/TPAMI.1986.4767851
    """
    if low > high:
        low, high = high, low
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

def imhist(ridge):
    edges = np.linspace(start=1./(255.*2), stop=1-(1./(255.*2)), num=254)
    edges = np.insert(edges, 0, 0)
    edges = np.insert(edges, 255, 1)
    x = np.histogram(np.where(ridge.ravel()>0, ridge.ravel(), 0), bins=edges)
    return(x)

def threshold_all(ridge_validated):
     T = np.zeros((2, 1))
     histogram, edges = imhist(ridge_validated)
     histogram = np.delete(histogram, 0)
     T[0, 0] = RosinThreshold(histogram)/256.
     ridge2 = ridge_validated.copy()
     ridge2[ridge_validated > T[0,0]] = 0
     histogram, edges = imhist(ridge2)
     histogram = np.delete(histogram, 0)
     T[1, 0] = RosinThreshold(histogram)/256.
     return T
	 
def threshold_segments(ridge_validated, segments):
    T = np.zeros((2, int(np.max(segments))+1))
    for seg in range(int(np.max(segments))+1):
        ridge_validated_trunc = \
            np.where(segments.ravel() == seg, ridge_validated.ravel(), 0)
        histogram, edges = imhist(ridge_validated_trunc)
        histogram = np.delete(histogram, 0)
        T[0, seg] = RosinThreshold(histogram)/256.
        ridge2 = ridge_validated_trunc.copy()
        ridge2 = np.where(ridge_validated_trunc.ravel() <= T[0, seg], \
                          ridge2.ravel(), 0)
        ridge2[ridge_validated_trunc > T[0,0]] = 0
        histogram, edges = imhist(ridge2)
        histogram = np.delete(histogram, 0)
        T[1, seg] = RosinThreshold(histogram)/256.
    return T

def apply_threshold_segments(ridge_validated, T, segments, dilate=0):
    crackmap = np.zeros(ridge_validated.shape)
    for seg in range(int(np.max(segments))+1):
        crackmap_seg = apply_hysteresis_threshold(ridge_validated, \
            T[0, seg], T[1,seg])
        crackmap_seg = np.where(segments == seg, crackmap_seg, 0)
        crackmap = crackmap_seg + crackmap
    if dilate:
        crackmap = ndimage.morphology.binary_dilation(crackmap)
    return crackmap
