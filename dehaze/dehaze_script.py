#!/usr/env/python3

import utils

utils.install_and_import("imageio", "imageio")
from utils import imageio

from onc.onc import ONC
import shutil
import os
import numpy as np
from scipy.stats import zscore
from scipy.ndimage.filters import minimum_filter as min_filter
import json

def atmosphere(img, img_dark, ratio):
    ind = img_dark >= img_dark.max()*(1-ratio) 
    atm = np.max(img[ind], axis=0)[np.newaxis,:] 
    return atm  

def transmission(img, at, omega, wsize):
    #t(x) = 1-omega min min(I^C(x) / A^c) 
    t = 1.0 - omega * dark_channel(img / at , wsize )
    return t[:,:,np.newaxis]

def dark_channel(img,wsize):
    #J_dark(x) = min_wsize ( min J^c(y) )
    j_min  = np.min(img, axis=2, keepdims=False)
    return min_filter(j_min, size=wsize)

def reconstruct( img, at, t, t_0):
    #J = I(x) - A / max(t, t_min) + A
    J = np.abs(img - at) / np.maximum(t_0, t) + at 
    return (1 - (J - J.min()) / (J.max() - J.min())) * 255

def preprocess(src):
    img = src.copy().astype(np.float32)
    img = zscore(img)
    img = np.clip(img, -1, 1)
    return (img + 1.) / 2.

def dehaze(src, opts):
    img = preprocess(src) if opts["preprocess"] else src.copy().astype(np.float32) 

    img_dark = dark_channel(img, opts["wsize"])
    at = atmosphere(img, img_dark, opts["ratio"])
    t = transmission(img, at, opts["omega"], opts["wsize"])
    if opts["refine"]:
        import cv2 as cv
        from cv2.ximgproc import guidedFilter
        t = guidedFilter(cv.cvtColor(src, cv.COLOR_BGR2GRAY), t, 5, 0.85)[:,:,np.newaxis]
    return reconstruct(img, at, t, opts["t_0"])

with open("params.json", "r") as f:
    params = json.load(f)

    onc_    = params["onc"]
    search_ = params["search"]
    opts    = params["dehaze"]

    from onc.onc import ONC
    onc = ONC(onc_["token"], 
              onc_["production"],
              onc_["showInfo"], 
              onc_["outPath"])


    dps = onc.getDataProducts(filters={
            'deviceCode' : search_["deviceCode"]})[0]

    query = {
        'dateFrom'          : search_["dateFrom"], 
        'dateTo'            : search_["dateTo"], 
        'dataProductCode'   : dps['dataProductCode'],
        'extension'         : dps['extension'],
        'deviceCode'        : search_["deviceCode"],
    }
    
    orders = onc.orderDataProduct(
                query, 
                onc_["maxRetries"], 
                onc_["downloadResultsOnly"], 
                onc_["includeMetadataFile"])
    
    downloaded = [ order['file'] for order in orders['downloadResults'] if order['downloaded'] ] 

    print("Downloaded: ", downloaded)
    outPath = onc_["outPath"]
    for tar_fn in downloaded:
        images = utils.untar(tar_fn, outPath, "bmp")
        
        print("Processing tarfile: ", tar_fn)
        processed = [ dehaze(imageio.imread(os.path.join(outPath, img)), opts) for img in images ]
         
        for i, img in enumerate(processed):
            print("Saving ", images[i])
            imageio.imwrite(os.path.join(outPath,images[i]), np.uint8(img))

        source_dir = os.path.dirname(images[0])
        utils.tar(tar_fn, outPath, source_dir)
        print("Removing soure directory", source_dir)
        shutil.rmtree(os.path.join(outPath, source_dir)) 
