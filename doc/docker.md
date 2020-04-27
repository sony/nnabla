# Docker workflow with NNabla

If you are familiar with Docker and have your own workflow with Docker, you might skip this guide.

## What is Docker

* [Docker](https://www.docker.com/)
* Description from Wikipedia:
```
Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels.
```

## Installation of Docker and NVIDIA Container Toolkit

* Docker: If your host OS is Ubuntu, see [this official instsallation guide](https://docs.docker.com/install/linux/docker-ce/ubuntu/). You can find getting started guides for other OSs too.
* NVIDIA Container Toolkit: Visit [this page](https://github.com/NVIDIA/nvidia-docker) and follow the installation instruction.

## Build a docker image for nnabla-examples

Clone and move to the repository root (if you haven't done it).

```shell
git clone git@github.com:sony/nnabla-examples.git
cd nnabla-examples/
```

The following command creates a Docker image for nnabla examples. 

```
PROXY_OPTS="--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}"  # Can be omitted
docker build ${PROXY_OPTS} -t local/nnabla-examples .
```

## Run scripts inside Docker container

### Define custom docker-run command to share files between host and container

Usually, we use `docker run` command to execute a command inside a Docker container (virtual environment) of a created Docker image.
By the default setting of `docker run`, it doesn't share any files (precisely speaking it shares some system files though) with host,
so you can not exchange files between host and container.
However, if you perform training in a Docker container,
you may want to have access to scripts and dataset located on your local file system, and to use them from the container,
and you may want to get training results or logs from the container as well.

An option `-v` or `--volume` of `docker run` enables you to mount your host directory onto a specified path in a container
so that you have an access to the host directory from the container.
However, it is not a sufficient solution.
If you write a file to host from the container, the created file will be owned by root user (which means you cannot modify it or even remove it from host),
because a command in a Docker container are executed as a user in the container (by default, it's root!).

The following script defines a handy wrapper shell command for `docker run`, which offers the following benefits:

* You can access files on host file system from container as if you are on host (home and current directory are mounted)
* You can execute commands as a current user on host (no owner issues)
* Some configurations on host such as proxies are propagated to container

You can copy the following scripts to your `~/.bashrc` to make it available permanently.


```shell
export DOCKER_COMMAND=docker
docker_run_user () {
    # Create user options
    tempdir=$(mktemp -d)
    getent passwd > ${tempdir}/passwd
    getent group > ${tempdir}/group
    user_opts="-v${HOME}:${HOME} -w$(pwd) -u$(id -u):$(id -g)"
    user_opts="${user_opts} $(for i in $(id -G); do echo -n ' --group-add='$i; done)"
    user_opts="${user_opts} -v ${tempdir}/passwd:/etc/passwd:ro -v ${tempdir}/group:/etc/group:ro"

    # Create volume mount options for user home and current directory
    mp_home=$(df -P $HOME | awk 'NR==2 {print $6}')
    mp_pwd=$(df -P $(pwd) | awk 'NR==2 {print $6}')
    mnt_opts=""
    if [ $mp_home != '/' ]; then
        mnt_opts="${mnt_opts} -v ${mp_home}:${mp_home}:rw"
    fi
    if [ $mp_pwd != '/' ]; then
        mnt_opts="${mnt_opts} -v ${mp_pwd}:${mp_pwd}:rw"
    fi

    # Create proxy options
    proxy_opts="-ehttp_proxy=${http_proxy} -ehttps_proxy=${https_proxy} -eftp_proxy=${ftp_proxy}"


    # Other otions
    misc_opts="--rm -ti"
    # Run docker run command with created options
    ${DOCKER_COMMAND} run ${user_opts} ${mnt_opts} ${proxy_opts} ${misc_opts} "$@"
}
```

**Example usages**:
```shell
docker_run_user local/nnabla-examples echo "Hello nnabla."
docker_run_user local/nnabla-examples -v /data:/data ls /data  # Suppose /data is on host, and additionally you want to mount it
```

### Run example script in Docker container 

The following script executes MNIST classification training on a container.


```shell
cd mnist-collection
docker_run_user --gpus 0 local/nnabla-examples python classification.py -c cudnn -d 0
```

**Note**:

* An appropriate version of CUDA driver must be installed to run with NVIDIA GPUs (with `--gpus` option).
