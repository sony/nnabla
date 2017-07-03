Run A Docker Image hosted on DockerHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker pull nnabla/nnabla
    docker run nnabla/nnabla python -c "import nnabla"

If it works, you should see

.. code-block:: bash

    2017-07-03 02:52:51,435 [nnabla][INFO]: Initializing CPU extension...

Build Docker Image from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone git@github.com:sony/nnabla.git
    cd nnabla
    docker build -t nnabla -f docker/Dockerfile .
    docker run nnabla python -c "import nnabla"

If it works, you should see

.. code-block:: bash

    2017-07-03 02:52:51,435 [nnabla][INFO]: Initializing CPU extension...
