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

To install the using the `pip` requirements file.  

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

## Client Library

The Client library is also available in Matlab and R.

[Oceans 2.0 API](https://wiki.oceannetworks.ca/display/O2A/Oceans+2.0+API+Home)

## Authors

* **Amanda Dash** - *Initial work* - [m3m3t](https://github.com/m3m3t)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
