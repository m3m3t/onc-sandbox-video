"""
This algorithm using a Gaussian Mixture Model and polynomial activation model
to label foreground and background objects in relation to novelty of motion captured
through pixel intensity changes
"""

import cv2 as cv
from cv2.ximgproc import guidedFilter
import numpy as np

"""
Helper enum class: 
  Pixels that exihibit high novelty or motion are set to 255
  Pixels that exihibit medium novelty or repetitive motion are set to 175
"""
class PixelEvent(object):
    NEW_MOTION  = 255
    OLD_MOTION  = 175
    ALL_OBJ     = (NEW_MOTION | OLD_MOTION) 

class PCH(object):
    def __init__(self, **kargs):
        #These parameters controls how fast/slow/much the pixels are labelled as
        #novel and can be adjusted.  These values are "general-purpose" and will work
        #for most cases.
        self._T_H             = kargs.pop("event_threshold", 175)
        self._T_M             = kargs.pop("event_type_threshold", 10) 
        self._learningRate    = kargs.pop("learning_rate", 0.1)
        
        #These can be changed, but tweaking the above values should be sufficient.
        self._accum_factor    = kargs.pop("accumulate_rate", 2.0) 
        self._decay_factor    = kargs.pop("decay_rate", 0.15) 
       
        #This does not currently work in the sandbox but can be used to improve speed of
        #of local testing if CUDA/numba is installed
        self._use_gpu         = kargs.pop("use_gpu", False)
        

        if kargs:
            raise TypeError("'{}' is an invalid keyword".format(kargs.popitem()[0]))            

        self.setup_matrix_functions()

    """Function sets up the activation and history matrixes depending on whether a 
       gpu is used or not.  These functions control how the foreground evolves
    """
    def setup_matrix_functions(self):
        if self._use_gpu:
            print("Using CUDA ...")
            from numba import vectorize 
            @vectorize(['uint8(int8,int8,float32)'], target='cuda')
            def activation(a,b,c):
                if c < 175: return 0
                if abs(a-b) > 10: return 255 
                return 175
            self.activation = activation

            @vectorize(['float32(float32)'], target='cuda')
            def adjust(t):
                return -2.*t*t*t + 3.*t*t

            self.adjust = adjust
        else:
            self.activation = np.vectorize(self._activation)
            self.adjust = np.vectorize(self._adjust)
        
    #Setters/Getters for matrices
    @property
    def T_matrix(self):
        return self.__T_matrix
    @T_matrix.setter
    def T_matrix(self, el):
        if type(el) is tuple:
            self.__T_matrix = np.zeros(el,dtype="float32")
            self.__T_matrix.fill(-6) 
        else:
            self.__T_matrix = el

    @property
    def D_matrix(self):
        return self.__D_matrix
    @D_matrix.setter
    def D_matrix(self,x):
        self.__D_matrix = x 

    @property
    def P_matrix(self):
        return self.__P_matrix
    @P_matrix.setter
    def P_matrix(self,el): 
        if type(el) is tuple:
            self.__P_matrix = np.zeros(el,dtype="float32")
        else:
            self.__P_matrix = el

    @property
    def E_matrix(self):
        return self.__E_matrix
    @E_matrix.setter
    def E_matrix(self,el): #(h,w)):
        if type(el) is tuple:
            self.__E_matrix = np.zeros(el,dtype="uint8") #Event matrix
        else:
            self.__E_matrix = el
   
    #This is the GMM background model. Underwater this is very noisy so 
    #we can't use it "as-is".
    @property
    def foreground(self):
        if self.__foreground is None:
            self.__foreground = cv.bgsegm.createBackgroundSubtractorMOG(
                                    history=self._gmmHistory 
                                    )
        return self.__foreground
    @foreground.setter
    def foreground(self,x):
        self.__foreground = x 
   
    #This functions controls how fast/slow the probability a pixel is novel
    #based on it's history.
    def _adjust(self, t):
        return -2.*t*t*t + 3.*t*t

    #This function sets a pixel as novel/kinda novel/or not novel based on 
    #threshold values.
    def _activation(self, a,b,c):
        return 0 if c < self._T_H \
                 else PixelEvent.NEW_MOTION \
                 if abs(a-b) > self._T_M \
                 else \
                 PixelEvent.OLD_MOTION

    #Update the current model state
    def update_model(self, frame_prev, frame_curr):
        self.E_matrix[:] = 0 

        #The GMM will give 255 for foreground pixels and 0 for background.
        #Foreground pixels increase, and background decrease by a set factor
        self.D_matrix = self.foreground.apply(
                                frame_curr,
                                self.D_matrix, 
                                learningRate=self._learningRate) 
        D = self.D_matrix.copy().astype("float32")
        D = D / 255 * self._accum_factor 
        D[D == 0] = -self._decay_factor
      
        #We add the latest evolution to our time matrix T and ensure
        #that we remain in the range [0,1] to ensure math. stability
        self.T_matrix = np.add(D, self.T_matrix)
        self.T_matrix = np.clip(self.T_matrix, 0, 1)

        #Calculate the probability of each pixel and run the activation function to label the pixels
        self.P_matrix = self.adjust(self.T_matrix) * 255
        self.E_matrix = self.activation(np.int8(frame_prev), np.int8(frame_curr), self.P_matrix)
   
        return guidedFilter(frame_curr, np.uint8(self.E_matrix), 5, 0.1)
       

    #Initialization function
    def initialize(self, frame_size, fps):
        self._frame_size = frame_size

        #How long the GMM remembers, this can be changed
        #I believe the default is 200
        self._gmmHistory = int(fps)   #1 second window
        
        self.foreground = None 
        self.D_matrix = None
        self.P_matrix = frame_size 
        self.E_matrix = frame_size 
        self.T_matrix = frame_size
    
