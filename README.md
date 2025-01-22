# Mozza

GStreamer plugin for smile transformation.

## Dependencies

The preferred option is to build from the Docker image created by `docker/Dockerfile.code` that comes with latest dependencies versions.

Alternatively you may (under a debian-based distributions, Buster being to be the minimum requirement):

```
sudo apt-get install libdlib-dev libopencv-dev libgstreamer-opencv1.0-0
```


## Build

```
meson build
meson configure -Dbuildtype=release build
ninja -C build
```

This will build both the GStreamer plugin and the `mozza-templater` tool.

You can either use the plugin as is, by providing the path to the build folder using the env variable GST_PLUGIN_PATH. Or you can install the plugin (though keep in mind that GStreamer doesn't search for plugins in the `/usr/local/` prefix by default:

```
meson install -C build -Dprefix=/usr[/local]
```

## Using mozza-templater

First, make sure you know where to find the `shape_predictor_68_face_landmarks.dat` model file.
It can be found online at [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](), or on debian distributions, it will have been provided by the libdlib-data package.

### Interactive mode

The primary use of this program is to generate the deformation vector used by the plugin.
Here is a simple example, defaulting to the computer webcam as a video source.

```
mozza-templater -m </path/toshape_predictor_68_face_landmarks.dat> -o <deformation_file.dfm>
```

Once started, select a frame with a neutral expression by pressing the `N` key, then do the same for a frame with a smile by pressing the `S` key. Finally, you can export the deformation file by pressing `O`.

### Batch mode

You can also create a deformation file from the command line by using the `-b` flag and provide two images of a neutral and a smiling face:

```
mozza-templater -b -m </path/to/shape_predictor_68_face_landmarks.dat> -n <neutral_img> -s <smile_img> -o <deformation_file.dfm>
```

## Running the plugin

| Property      | Type    | Description                                                                                                                                                                                                                              | Default                                                 |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `shape-model` | String  | Path to shape predictor model file. Can be found online at [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](), or on debian distributions, it will have been provided by the libdlib-data package.                      | `/usr/share/dlib/shape_predictor_68_face_landmarks.dat` |
| `deform`      | String  | Path to the deformation file                                                                                                                                                                                                             | `./default.dfm`                                         |
| `drop`        | Boolean | Whether or not to drop the frames where we failed to detect any faces                                                                                                                                                                    | `false`                                                 |
| `alpha`       | Float   | Smile multiplicator: at its default value of one, the template smile deformation is applied as is. A negative value will apply a frown.                                                                                                  | `1.0`                                                   |
| `face-thresh` | Float   | Optional confidence threshold that can be tweaked if the face detection algorithm struggles with false positives or fails to detect valid faces.                                                                                         | `0.25`                                                  |
| `overlay`     | Boolean | enable/disable debug overlay.                                                                                                                                                                                                            | `false`                                                 |
| `beta`        | Float   | One euro jitter filter beta. If there is an issue with lagging landmarks during high speed movements, increase beta. For more detailed informations on tuning the One Euro Filter, visit [http://cristal.univ-lille.fr/~casiez/1euro/](http://cristal.univ-lille.fr/~casiez/1euro/) | `0.1`                                                   |
| `fc`          | Float   | One euro jitter filter frequency cutoff. If there is an issue with jittering landmarks at low speed, decrease fc.                                                                                                                        | `5.0`                                                   |
| `user-id`     | String  | User ID used to tag tracking logs.                                                                                                                                                                                                       | `0`                                                     |

Example pipelines:

```
GST_PLUGIN_PATH="$GST_PLUGIN_PATH:./build" GST_DEBUG=2 gst-launch-1.0 filesrc location=<test_video_file.mp4> ! decodebin ! videoconvert ! mozza shape-model=</path/to/shape_predictor_68_face_landmarks.dat> deform=</path/to/deformation file.dfm> [drop=<true|false>] [alpha=[-10; 10]] ! videoconvert ! autovideosink
```

```
GST_PLUGIN_PATH="$GST_PLUGIN_PATH:./build" GST_DEBUG=2 gst-launch-1.0 filesrc location=./data/in/test.mkv ! matroskademux ! avdec_h264 ! videoconvert ! mozza deform=./data/out/test.dfm alpha=1 overlay=true ! videoconvert ! x264enc ! matroskamux ! filesink location=./data/out/test.mkv
```

## Use mozza with Docker

To provide with a multi-platform solution that does not require to recompile mozza, one way is to run mozza binaries (`mozza-templater` and the GStreamer plugin) within Docker containers.

The process is to install Docker (or Docker Desktop depending on your platform) and then follow the instructions in the next paragraph [Rely on ducksouplab/mozza](#rely-on-ducksouplabmozza) so that you already have a Docker image to create your mozza containers.

The following paragraph [Rebuild with Docker](#rebuild-with-docker) explains how to build a Docker image such as `ducksouplab/mozza`.

### Rely on ducksouplab/mozza

1. Get latest ducksouplab/mozza

Login to `ducksouplab` account (for the time being the image is private) and pull image:

```
docker login
docker pull ducksouplab/mozza:latest
```

Redo this step whenever a new image has been published.

2. Prepare environment

Choose a working directory and create folders to share files between your host computer and the mozza container:

```
mkdir -p data/in data/out
```

In steps 3, 4, 5 below, commands are in the form:

```
docker run -v "$(pwd)"/data:/data ducksouplab/mozza:latest <container_command>
```

The idea is: use docker to run `<container_command>` in a container that comes with mozza installed (thanks to the `ducksouplab/mozza:latest` prebuilt image), and by binding my local `data` folder to `/data` in the container.

Then in `<container_command>`: whatever files are read from `/data/in` or written to `/data/out` will be mapped to the ones in the `data/in` and `data/out` local folders we've just created.

It's possible to change the `-v "$(pwd)"/data:/data` setting to mount a different volume from your computer, but you may keep the last `:/data` part that specifies a folder that actually exists (and is empty) in the container.

3. Use templater

Create a deformation (.dfm) file with `mozza-templater`:

```
docker run -v "$(pwd)"/data:/data ducksouplab/mozza:latest mozza-templater -b -m /usr/share/dlib/shape_predictor_68_face_landmarks.dat -n /data/in/neutral.png -s /data/in/smile.png -o /data/out/test.dfm
```

4. Process an image

Use the deformation file and process an image:

```
docker run --env GST_DEBUG=2 -v "$(pwd)"/data:/data ducksouplab/mozza:latest gst-launch-1.0 filesrc location=/data/in/test.png ! decodebin ! videoconvert ! mozza deform=/data/out/test.dfm alpha=1 ! videoconvert ! jpegenc ! filesink location=/data/out/transformed.jpg
```

5. Process a video

Create a `raw.mkv` file and put it in `data/in, then process it:

```
docker run --env GST_DEBUG=2 -v "$(pwd)"/data:/data ducksouplab/mozza:latest gst-launch-1.0 filesrc location=/data/in/raw.mkv ! decodebin ! videoconvert ! mozza deform=/data/out/test.dfm alpha=1 ! videoconvert ! x264enc ! mp4mux ! filesink location=/data/output.mp4
```

6. A note on GST_DEBUG

Whenever you use GStreamer, for instance with `gst-launch-1.0`, you may specify a GStreamer debug log level. The suggester `GST_DEBUG=2` should give enough information if a simple problem prevents the pipeline to work (for instance due to a missing file or plugin, or incompatible media types).

## Rebuild with Docker

1. Download dlib files

Download dlib source once (used to build the Docker image):

```
mkdir -p docker/deps/dlib
curl http://dlib.net/files/dlib-19.24.tar.bz2 --output docker/deps/dlib/dlib-19.24.tar.bz2
```

2. Create image

Under macOS or Windows you may have to increase the RAM available to Docker Desktop for the build process to work.

```
docker build --progress=plain -f docker/Dockerfile.build -t mozza:latest .
```

To recreate it without caches:

```
docker build --progress=plain -f docker/Dockerfile.build -t mozza:latest .
```

If you want to push it to `ducksouplab/mozza`:

```
docker login
docker tag mozza:latest ducksouplab/mozza:latest
docker push ducksouplab/mozza:latest
```

3. Compile for ARM architectures

docker build --progress=plain -f docker/ArmDockerfile.build -t mozza:arm_latest .
docker tag mozza:arm_latest ducksouplab/mozza:arm_latest
docker push ducksouplab/mozza:arm_latest


4. Create image from cuda ubuntu

```
docker build --progress=plain -f docker/Dockerfile.cuda.build -t mozza:cuda-latest .
```

5. Run mozza templater and pipeline from host

Create a folder that will be used to store mozza generated files:

```
mkdir -p data/out
```

Use the templater to create a deformation (.dfm) file:

```
docker run -v "$(pwd)"/data:/data ducksouplab/mozza:latest mozza-templater -b -m /usr/share/dlib/shape_predictor_68_face_landmarks.dat -n /data/in/neutral.png -s /data/in/smile.png -o /data/out/test.dfm
```

Use the deformation file and process an image:

```
docker run -v "$(pwd)"/data:/data ducksouplab/mozza:latest gst-launch-1.0 filesrc location=/data/in/test.png ! decodebin ! videoconvert ! mozza deform=/data/out/test.dfm alpha=1 ! videoconvert ! jpegenc ! filesink location=/data/out/transformed.jpg
```

## Check the tutorials
Check a tutorial that runs you through each step to use mozza in local in [this link](https://github.com/ducksouplab/mozza/blob/main/tutorials/Use_mozza_in_local.md)

## Additional: run mozza by entering container

Enter the container:

```
docker run -it --rm -v "$(pwd)"/data:/data ducksouplab/mozza:latest bash

```

_From now on, you're in the container shell_

```
mozza-templater -b -m /usr/share/dlib/shape_predictor_68_face_landmarks.dat -n /data/in/neutral.png -s /data/in/smile.png -o /data/out/test.dfm

# inspect by plugin file
gst-inspect-1.0 build/libgstmozza.so

# inspect by plugin name (since libgstaltavoce.so is in GST_PLUGIN_PATH)
gst-inspect-1.0 mozza


gst-launch-1.0 filesrc location=/data/in/test.png ! decodebin ! videoconvert ! mozza deform=/data/out/test.dfm alpha=1 ! videoconvert ! jpegenc ! filesink location=/data/out/transformed.jpg


```

Exit container:

```
exit
```
