from onc.onc import ONC

import requests
import cv2 as cv
import numpy as np
import os
import editdistance
from matplotlib import pyplot as plt

from summarize import utils
from summarize.pch import PCH
from summarize.utils import Timer

class VideoDownloader(object):
    video_formats = ['mp4', 'mpg', 'avi' ]
    image_formats = ['bmp', 'jpg', 'gif', 'png', 'jpeg', 'zip']
    valid_formats = video_formats + image_formats 

    def __init__(self, token, data_dir="data", 
                              dateFrom=None,
                              dateTo=None,
                              production=True, 
                              verbose=False):
        self._onc = ONC(token, 
                        production=production,
                        showInfo=verbose,
                        outPath=data_dir)
        self._data_dir = data_dir
        self._verbose  = verbose

        self.filters = {}
        if dateFrom is not None:
            self.filters = {'dateFrom' : dateFrom }
        if dateTo is not None:
            self.filters = {'dateTo' : dateTo }

        self._urls = None
        self._query = None
        self.__dataProduct = None

        """
        TODO: the onc module should have a __version__ flag
        if verbose:
            import onc
            print("ONC API version: ", onc.onc.__version__)
        """

    @property
    def dataProduct(self):
        return self.__dataProduct
    @dataProduct.setter
    def dataProduct(self, filters):
        self.__dataProduct = [ d for d in self._onc.getDataProducts(
                    filters=filters) if d['extension'] in self.valid_formats ]
         
    @property
    def filters(self):
        return self.__filters
    @filters.setter
    def filters(self, _filters):
        if hasattr(self, 'filters'):
            self.__filters = {**self.__filters, **_filters}
        else:
            self.__filters = _filters 

    def findDataProducts(self, deviceCode, video_only=False, image_only=False):
        self.filters = {'deviceCode': deviceCode}
        locations = self._onc.getLocations(
                filters={ k:v for k,v in self.filters.items() if k != 'deviceId'})
        
        if self._verbose:
            print("No of locations: ", len(locations))

        #TODO: filter by video extension
        dataProducts = [] 
        filters = { k:v for k,v in self.filters.items() if 'date' not in k }
 
        for i,location in enumerate(locations):
            locationCode = location['locationCode']
            hasDeviceData = location['hasDeviceData']
            hasPropertyData = location['hasPropertyData']

            #need deviceCategoryCode
            if hasDeviceData:
                cats = self._onc.getDeviceCategories(
                        filters={'locationCode' : locationCode})
                cats = [ c for c in cats if 'CAMERA' in c['deviceCategoryName'].upper() ]
                for cat in cats:        
                    deviceCategoryCode = cat['deviceCategoryCode']
                    filters['locationCode'] = locationCode 
                    filters['deviceCategoryCode'] = deviceCategoryCode 
                    filters['deviceCode'] = deviceCode
                    for d in self._onc.getDataProducts(filters=filters):
                        if video_only and d['extension'] not in self.video_formats:
                            continue
                                              
                        if image_only and d['extension'] not in self.image_formats:
                            continue
 
                        #There is a bug where this doesn't always retrieve valid results 
                        self.dataProduct = { 'deviceCode' : deviceCode,
                                             'extension' : d['extension'] }

                        if len(self.dataProduct) and \
                           self.dataProduct[0]['dataProductCode'] == d['dataProductCode']:
                            d['locationCode'] = locationCode
                            d['deviceCategoryCode'] = deviceCategoryCode
                            d['locationName'] =  location['description']
                            d['deviceCode'] = deviceCode
                            dataProducts.append(d)
                       
            elif hasPropertyData:
                raise NotImplemented('hasPropertyData requires propertyCode')
            

        return dataProducts

    def searchForDevice(self, deviceName, deviceId=None):
        best_match = {}
        filters = self.filters.copy()
        if deviceId is not None:
            filters['deviceId'] = deviceId

        avail_devices = self._onc.getDevices(filters=filters)
        if deviceId is not None:
            return avail_devices
            
        for i,device in enumerate(avail_devices):
            _name = device['deviceName'].upper()
            if 'VIDEO' in _name or 'CAMERA' in _name:
                s = editdistance.eval(_name, deviceName)
                best_match[i] = s
    
        best_match = [(avail_devices[k]) for k in sorted(best_match, key=best_match.get, reverse=True) ]

        return best_match

    def print_urls(self):
        print("  {}\n".format(*self._urls))


    def retrieveProductInfo(self, query):
        params = ['dateFrom', 'dateTo', 'extension', 'dataProductCode',
                  'locationCode', 'deviceCategoryCode']
 
        self._query = {**self.filters, **query} 
        self._query = { k:v for k,v in self._query.items() if k in params }
        
        assert len(params) == len(self._query.keys())
        
        res = self._onc.requestDataProduct(self._query)
        
        if res is None:
            print("Query did not yield results, try changing your `dateFrom` and `dateTo` parameters")
            return None, 0

        self._file_size = res['fileSize'] 
        self._num_files = res['numFiles'] 

        #TODO: includeMetadataFile=False option would be nice here
        self._urls = self._onc.getDataProductUrls(self._query)
        self._urls = [ url for url in self._urls if 'index=meta' not in url ]

        #BUG: numFiles is not he correct number (ends up the urls inlucde the meta file) so not a bug? 
        assert self._file_size == res['compressedFileSize']

        return self._urls, res['fileSize']

    def downloadVideo(self, index, output_dir=None, keep_original=True):

        if not (0 <= index < len(self._urls)):
            raise IndexException("Index must be a valid array index.")

        if output_dir is None:
	        output_dir = self._data_dir

        url = self._urls[index]
        if self._verbose:
            print("Downloading ...",url)

        from tqdm import tqdm
        out_fn = ""
        rsp = requests.get(url, stream=True)
        if rsp.ok and rsp.status_code == 200: 
            if 'Content-Disposition' in rsp.headers.keys():
                content = rsp.headers['Content-Disposition']
                filename = content.split('filename=')[1]
                ext = filename.split(".")[-1]

            chunk_size = 8192 #min(self._file_size // 10, 8192)
            out_fn = "{}/{}".format(output_dir, filename)
            if not os.path.isfile(out_fn):
                if self._verbose:
                    print("Saving ", out_fn)
                
                with open(out_fn, "wb") as f:
                    for chunk in tqdm(rsp.iter_content(chunk_size = chunk_size)):
                        f.write(chunk)
            else:
                print("{} already exists.".format(filename))
        else:
            print("ERROR: bad stream.")

        return out_fn

    def summarize(self, video_fn):
        frames = {}
        try:
            video, props = utils.load_video(video_fn)
        except:
            print("Unable to open ", video_fn)
            return {} 
        
        h,w = props['frame_size']
        fps = props['fps']
        if w > 800 or h > 600:
            h,w = 600,800 
        frame_size = (w,h)
        
        pch = PCH()
        pch.initialize((h,w), fps)

        frames = self._runVideo(video, frame_size, int(fps), pch)
        video.release()
        return frames 

    def _runVideo(self, video, frame_size,  fps, pch, show=False):
        w,h = frame_size
        frame_color = None
        _, frame_color_prev = video.read()
        
        frame_color_prev = cv.resize(frame_color_prev,frame_size, interpolation = cv.INTER_CUBIC)
        
        frame_num = 0 
        frames = {} 
        while True:
            _, frame_color = video.read()

            if frame_color is None:
                print("End of video")
                break
            
            frame_num += 1
            frame_color = cv.resize(frame_color,frame_size, interpolation = cv.INTER_CUBIC)
            result = pch.update_model(frame_color_prev, frame_color)
            
            result[result < 175] = 0
            new_motion = np.count_nonzero(result[result > 225]) 
            old_motion = np.count_nonzero(result[result < 225])
            frames[frame_num] = [new_motion,old_motion]
            
            if frame_num % (15*fps) == 0:
                print("Processed {} secs".format(frame_num / fps))
            
            if show: 
                plt.hist(result.ravel(), 256, [0,256]); plt.pause(0.001)
                cv.imshow("frame", frame_color)
                cv.imshow("pch", cv.applyColorMap(result, 
                                    cv.COLORMAP_JET))
                cv.waitKey(1)
            
            frame_color_prev = frame_color

        return frames 
