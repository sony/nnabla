# NNabla Dockers

## Image Tags Hosted on DockerHub

The available tags are as following.

| Tag    | Python | Dockerfile location |
| ------ |:------:|:------------------- |
| latest | 3.5    | py3/                |

A docker image can be executed as below.

```
docker run <options> nnabla/nnabla:<tag> <command>
```

### Tutorial image

This image contains the latest NNabla and its tutorials on Python3.
The following command runs a jupyter server listening 8888 on the host OS.

```
docker run --rm -p 8888:8888 nnabla/nnabla:tutorial
```

You can connect the server with your browser by accessing
`http://<Host OS address>:8888`. The login password is `nnabla`.

After logging in, the page show you a list of tutorials as Jupyter notebook `.ipynb` files.

### Android: `Dockerfile`
Dockerfile for building NNabla using Android NDK is present at docker/development/Dockerfile.android in the NNabla repository.
Use the following command to build the Docker image.  
The following must be build at the root directory of NNabla.  
```
docker build -t nnabla-android --build-arg http_proxy=http://${proxy}:${port}/ --build-arg https_proxy=http://${proxy}:${port}/ -f docker/development/Dockerfile.android ../../
```
The above build will create docker image with tag nnabla-android:latest.

Use following command to run the docker.  
```
docker run -v $(pwd):$(pwd) -w$(pwd) -u $(id -u):$(id -g) -e HOME=/tmp nnabla-android:latest ./build-tools/android/build_nnabla.sh -p=android-26 -a=arm64 -n=/usr/local/src/android-ndk -e=arm64-v8a
```
#### Note:
Use the following command if you are behind the proxy.
```
docker run -ehttp_proxy=${http_proxy} -ehttps_proxy=${https_proxy} -eftp_proxy=${ftp_proxy} -v $(pwd):$(pwd) -w$(pwd) -u $(id -u):$(id -g) -e HOME=/tmp nnabla-android:latest ./build-tools/android/build_nnabla.sh -p=android-26 -a=arm64 -n=/usr/local/src/android-ndk -e=arm64-v8a
```

Please refer [here](https://github.com/sony/nnabla/tree/master/doc/build/build_android.md) for detailed instruction on building NNabla using Android NDK.  



