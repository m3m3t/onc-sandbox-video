# onc-sandbox-video
This project gives two Python 3 use cases for video and image process for the [ONC Oceans 2.0](https://data.oceannetworks.ca) API and Sandbox.  An Oceans 2.0 account is required (it's free) to obtain an API token to run the examples.  The token can be obtained by creating an account and accessing the [Web Services API tab](https://data.oceannetworks.ca/Profile).  Update the `params.json` file(s) with your token to run the use cases.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

__Python 3 modules__
* onc               - Oceans 2.0 Python client library
* OpenCV 3.x        - OpenCV Computer Vision Library          
* OpenCV contrib    - OpenCV extra modules
* imageio           - Image and video processing (faster than the OpenCV)
* sklearn           - Machine learning libraries
* numpy             - Advance python matrix manipulation
* numba             - CUDA GPU bindings (optional)
* pip               - Python package installer (optional)

__Libraries__
* ffmpeg            - Video processing
* cuda-toolkit      - GPU processing (optional)

### Installing

To install the using the `pip` requirements file.  Each use case has it's own `requirements.txt` file which also includes the optional modules.

```
pip install -r requirements.txt
```

This will install the opencv-python* packages but not any additional libraries that are required and possibly override any custom installations.  To test if the installation was successful, from the python 3 console:

```
import cv2
cv2.__version__
```

**NOTE**: You can install ffmpeg using `imageio` API if you are working in `conda` or `virtualenv`:
```
imageio.plugins.ffmpeg.download()
```

See individual use case REAMDEs for testing and deploying locally.

## Deployment

[Deploying in the Sandbox](https://drive.google.com/open?id=1eVfsFQbJX2QYvnP3pKdwGbRFLl6aowwX)

**Note**:  Select the `params.json` and the `utils.py` files as well when loading the scripts described in the video.

## Client Library

The documentation for the Oceans 2.0 API is available [here](https://wiki.oceannetworks.ca/display/O2A/Oceans+2.0+API+Home).  Additional scripts and instructions on exploring and discovering available videos available (in Matlab and R as well).

There will be cases when the Sanbox does not have a particular python module.  A workaround is to import the module at runtime (an example of this can be found in `dehaze_script.py`.  This is an example of function:

```python
import importlib

def install_and_import(module,package):
    try:
        importlib.import_module(module)
    except ImportError:
        import pip
        if pip.__version__.startswith('10'):
            from pip._internal import main as pipmain
        else:
            from pip import main as pipmain 
        
        pipmain(['install', '--target=.', package])
    finally:
        globals()[module] = importlib.import_module(module)

install_and_import("skimage", "scikit-image")
skimage.__version__
```

## Authors

* **Amanda Dash** - *Initial work* - [m3m3t](https://github.com/m3m3t)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
