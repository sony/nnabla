# WaveNet

This is a NNabla implementation of the [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).

Currently this example has not completed yet.  
You can see current implementation progress in "Supported Features" section below.

Hyper parameters are defined in [config.py](./config.py).  
Some values (like output channel of each layers, layer stack size, and so on) are not descrived in detail in the original paper.  
We use same values of [https://github.com/ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet) as reference.

## Dataset

We use LibriSpeech ASR corpus in this example.  
You can download dataset from [here](http://www.openslr.org/12/).


## Requirments
### Python environment
You can set up python dependencies from [requirements.txt](./requirements.txt):

```bash
pip install -r ./requirements.txt
```
Note that this requirements.txt dose not contain `nnabla-ext-cuda`.
If you have CUDA environment, we highly recommend to install `nnabla-ext-cuda` and use GPU devices.
See [NNabla CUDA extension package installation guild](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

### System environment
In this example, We use [PySoundFile](https://pysoundfile.readthedocs.io/en/latest/) to load FLAC format file
(This is also included in requirements.txt).
To use PySoundFile, you have to install [libsndfile](http://www.mega-nerd.com/libsndfile/) on your environment.

In the case of Ubuntu, you can install libsndfile by apt-get. Try:
```bash
apt-get install libsndfile-dev
```
If you use other OSs, see official documents to install libsndfile.

## Train
```bash
python train.py --device-id 0 \
                --context "cudnn" \
                --data-dir /path/to/LibriSpeech/ \
                --use-speaker-id
```

# Supported features
- [x] Train wavenet
- [x] Global conditioning (speaker embedding)
- [ ] Local conditioning
- [ ] Fast inference

# References

1. https://deepmind.com/blog/wavenet-generative-model-raw-audio/
2. [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)