# Use Mozza in Local and offline using Docker

This is a tutorial to run you through the steps to use Mozza with python.

Install Docker Hub:
- https://docs.docker.com/docker-hub/quickstart/

Mozza code is here : https://github.com/ducksouplab/mozza

Create a new folder called YOUR_FOLDER

Download the `shape_predictor_68_face_landmarks.dat` model file. It can be found online at [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](https://github.com/ducksouplab/mozza/blob/main), or on debian distributions, it will have been provided by the libdlib-data package.

Check the "Use mozza with Docker" part of the documentation [here](https://github.com/ducksouplab/mozza). Make sure you can login to docker and pull mozza docker image:
```
docker login
docker pull ducksouplab/mozza:latest
```

Create a new conda environement for mozza:
```
conda create --name mozza python=3.9 ipython jupyter
```

Clone the python wrapper repository : 
```
git clone https://github.com/Pablo-Arias/STIM.git
```

Add the python wrapper to the path of your environment (replace ABSOLUTE_PATH_TO_YOUR_FOLDER with your actual path):
```
conda develop ABSOLUTE_PATH_TO_YOUR_FOLDER/stim  
```

Crete a folder to transfer data:
```
mkdir -p data/in data/out
```

Make sure you have the files:
```
source = "neutral.png" # A neutral face to transform
def_file = "smile10.dfm" # Deformation file to use
```

Create a new python script and test the following image manipulation script:
```
from mozza_wrapper import transform_img_with_mozza, transform_video_with_mozza

# transform one image with mozza
container_folder = "ABSOLUTE_PATH_TO_YOUR_FOLDER/data" # replace with the folder t
source = "neutral.png"
target = "smile.png"
def_file = "smile10.dfm"
transform_img_with_mozza(container_folder, source, target, wait=True, deformation_file=def_file, alpha=1, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0)
```

Or the following video manipulation script (add a video_in.mp4 file):
```
container_folder = "ABSOLUTE_PATH_TO_YOUR_FOLDER/data"
source = "video_in.mp4"
target = "video_out.mp4"
def_file = "smile10.dfm"
transform_video_with_mozza(container_folder, source, target, wait=True, deformation_file=def_file, alpha=2.5, face_thresh=0.25 , overlay=False , beta=0.1, fc=5.0)
```

Play with parameters. Parameters files are described here : https://github.com/ducksouplab/mozza?tab=readme-ov-file#running-the-plugin



