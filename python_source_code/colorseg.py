import numpy as np
from scipy import ndimage
from sklearn import cluster
from PIL import Image

def fgaussian(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    credits to ali_m (stackoverflow.com)
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def rgb2lab(image):

    RGB = np.array(image) / 255.0
    RGB = 100*((RGB>0.04045)*((RGB+0.055)/1.055)**2.4+(RGB<=0.04045)*(RGB/12.92))
    
    X = RGB [:,:,0] * 0.4124 + RGB [:,:,1] * 0.3576 + RGB [:,:,2] * 0.1805
    Y = RGB [:,:,0] * 0.2126 + RGB [:,:,1] * 0.7152 + RGB [:,:,2] * 0.0722
    Z = RGB [:,:,0] * 0.0193 + RGB [:,:,1] * 0.1192 + RGB [:,:,2] * 0.9505
    X = X / 95.047 
    Y = Y / 100.0
    Z = Z / 108.883
    
    XYZ = np.append(X[:,:,np.newaxis],np.append(Y[:,:,np.newaxis],Z[:,:,np.newaxis],2),2)
    XYZ = (XYZ > 0.008856)*(XYZ**(1/3.0)) + (XYZ<=0.008856)*(16.0/116+7.787*XYZ)
    
    L = ( 116 * XYZ[ :,:,1 ] ) - 16
    a = 500 * ( XYZ[ :,:,0 ] - XYZ[ :,:,1 ] )
    b = 200 * ( XYZ[ :,:,1 ] - XYZ[ :,:,2 ] )

    Lab = np.append(L[:,:,np.newaxis],np.append(a[:,:,np.newaxis],b[:,:,np.newaxis],2),2)

    return Lab

def generate_segments(im, k, scale):
    im_w, im_h = im.size
    img_w = int(im_w * scale)
    img_h = int(im_h * scale)

    img_tn = im.resize((img_w, img_h))
    img_tn = np.array(img_tn)[:,:,0:3]
    h = fgaussian((9,9), 3)
    img_blurred = np.zeros(img_tn.shape)
    for i in range(3):
        img_blurred[:,:,i] = ndimage.convolve(img_tn[:,:,i], -h)
    img_blurred = img_blurred.astype(float)

    img_blurred = rgb2lab(img_blurred)

    c0 = img_blurred[:,:,0].flatten()[:,np.newaxis]
    c1 = img_blurred[:,:,1].flatten()[:,np.newaxis]
    c2 = img_blurred[:,:,2].flatten()[:,np.newaxis]
    lab = np.concatenate((c0,c1,c2), 1)

    clusterer = cluster.KMeans(n_clusters=k, algorithm='elkan')
    clustLabel = clusterer.fit_predict(lab)

    segments = clustLabel.reshape([img_h, img_w])
    segments = Image.fromarray(segments.astype('uint8'))
    segments = segments.resize((im_w, im_h))
    segments = np.array(segments)

    return segments






