# NNabla Dockers

## Image Tags Hosted on DockerHub

### Latest (default): `nnabla/nnabla`

This image contains the latest nnabla Python package working with Python3.

```
docker run -it --rm nnabla/nnabla
```

### Python2: `nnabla/nnabla:py2`

This image contains the latest nnabla Python package working with Python2.

```
docker run -it --rm nnabla/nnabla:py2
```

### Tutorial: `nnabla/nnabla:tutorial`

This image contains the latest NNabla and its tutorials on Python3.
The following command runs a jupyter server listening 8888 on the host OS.

```
docker run --rm -p 8888:8888 nnabla/nnabla:tutorial
```

You can connect the server with your browser by accessing
`http://<Host OS address>:8888`. The login password is `nnabla`.

## Dockerfiles for Developers

### Dev: `dev/Dockerfile`

Dockerfile used to create an image containing requirements for
building NNabla C++ libraries and Python package.

This must be build at the root directory of nnabla.

```
docker build -t local/nnabla:dev -f dev/Dockerfile ../
```

### Doc: `doc/Dockerfile`

Dockerfile used to create an image containing requirements for
building NNabla C++ libraries and Python package, as well as
the Sphinx documentation.

This must be build at the root directory of NNabla.

```
docker build -t local/nnabla:doc -f doc/Dockerfile ../
```

### Dist: `dist/Dockerfile`

Dockerfile for creating an image for building Python package wheels
for many linux distributions.

```
docker build -t local/nnabla:dist dist
```

TODO: Write more details.
