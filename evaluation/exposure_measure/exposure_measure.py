# *
# Python script to calculate the exposure measure provided by Eilertsen et al.
# Link to paper: https://www.cl.cam.ac.uk/~rkm38/pdfs/eilertsen2017tmo_star.pdf
# * #

import numpy as np
import cv2


def calcExposureMeasure(image):
    # calculate exposure measure for one given frame
    
    # calculate histogram of the image
    hist, bins = np.histogram(image, 256, [0, 255])
    # fraction of under-exposed pixels
    under_exp_pix = sum(hist[0:int(256*0.02)])/sum(hist) * 100
    # fraction of over-exposed pixels
    over_exp_pix = sum(hist[int(256*0.95):])/sum(hist) * 100
    
    return under_exp_pix, over_exp_pix


def exposure(pathToToneMappedImages):
    # calcuate exposure measure for all (already tone mapped!) frames in given path
    
    sequence_length = len(pathToToneMappedImages)
    under_exp_pix = 0
    over_exp_pix = 0

    for image in pathToToneMappedImages:
        # read tone mapped (TM) image as grayscale image
        toneMappedImage = cv2.imread(str(image), 0)
        
        curr_under_exp_pix, curr_over_exp_pix = calcExposureMeasure(toneMappedImage)

        # fraction of under-exposed pixels
        under_exp_pix += curr_under_exp_pix

        # fraction of over-exposed pixels
        over_exp_pix += curr_over_exp_pix

    # calculate average of under- and over-exposed pixels
    under_exp = under_exp_pix/sequence_length
    over_exp = over_exp_pix/sequence_length

    return under_exp, over_exp


