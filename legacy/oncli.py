#!/usr/bin/env python3

import argparse as ap
import pprint as pp
import editdistance
import csv
import cv2 as cv
from legacy.downloader import VideoDownloader
import json
import numpy as np
import os
import sys

def getUserInput(query, array):
    if len(array) == 0: 
        print("No results found.")
        sys.exit(1)

    key = 0
    if len(array) > 1:
        for i,s in enumerate(array):
            print("{}:".format(i))
            pp.pprint(s)
            print()

        key = int(input(query))
        print()
    return key

def reduceVideo(index):
    filename = downloader.downloadVideo(index) 
    json_fn = filename + ".npy"
    if os.path.isfile(json_fn):
        print("File already exists: ", json_fn)
        return
    framelist = downloader.summarize(filename)
    np.save(json_fn, framelist)

def processRequest(field, device_id, dateFrom, dateTo, anns, downloader, interactive):
    print("Search for: {}".format(field))
    matches = downloader.searchForDevice(field, device_id)

    key = getUserInput("Select device: ", matches if interactive else matches[:1])
    dps = downloader.findDataProducts(matches[key]['deviceCode'], video_only=True)
    key = getUserInput("Select data product: ", dps if interactive else dps[:1])
    
    dps[key]['dateFrom'] = dateFrom
    dps[key]['dateTo'] = dateTo

    urls, totalSize = downloader.retrieveProductInfo(dps[key])
    if totalSize == 0:
        print("No results returned.")
        return

    #TODO: use pickle instead of np.save
    if interactive:
        key = getUserInput("Select url: ", urls)
        reduceVideo(key) 
    else:
        for key in range(0,len(urls)):
            print("URL: ", urls[key])
            reduceVideo(key) 

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Download video files from Ocean 2.0")
    parser.add_argument("--token", default="")
    parser.add_argument("--device_name", default=None) 
    parser.add_argument("--device_id", default=None)
    parser.add_argument("--dateFrom", default='2016-12-23T00:00:00.000Z')
    parser.add_argument("--dateTo", default='2016-12-23T23:00:00.000Z')
    #TODO: make the dateFrom/To required 
    parser.add_argument("--batch_file", default=None) 
    parser.add_argument("--data_dir", default="data/test")
    
    parser.add_argument("--video_only", action="store_true") 
    parser.add_argument("--image_only", action="store_true") 
    parser.add_argument("--verbose", action="store_true") 
    
    args = parser.parse_args()

    if args.batch_file:
        params = json.load(open(args.batch_file, "r"))        
    else:
        params = [{'device_id' : args.device_id, 
                   'device_name' : args.device_name,
                   'dateFrom' : args.dateFrom,
                   'dateTo' : args.dateTo,
                   'annotations' : []}]

    #TODO: if data_dir does not exist, make it
    downloader = VideoDownloader(args.token, 
                                 data_dir=args.data_dir)
                                 #dateFrom=args.dateFrom,
                                 #dateTo=args.dateTo)

    interactive = (args.batch_file is None)
    print("Interactive mode: ", interactive)
    for p in params:
        processRequest(p['device_name'], p['device_id'], p['dateFrom'], p['dateTo'], p['annotations'], downloader, interactive)

    print("All done!")
