# onc-sandbox-video
This project gives two Python 3 use cases for video and image process for the [ONC Oceans 2.0](https://data.oceannetworks.ca) API and Sandbox.  An Oceans 2.0 account is required (it's free) to obtain an API token to run the examples.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* onc
* OpenCV 3.x 
* OpenCV contrib
* imageio
* sklearn
* ffmpeg 

### Installing

To install the using the `pip3` requirements file.  

```
pip install -r requirements.txt
```

**NOTE**: You can install ffmpeg using `imageio` API:
```
imageio.plugins.ffmpeg.download()
```

This will install the opencv-python* packages but not any additional libraries and possibly override any custom installations.  To test if the installation was successful, from the python 3 console:

```
import cv2
cv2.__version__
```

See individual use case REAMDEs for testing and deploying locally.

## Deployment

Add additional notes about how to deploy this on a live system

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
