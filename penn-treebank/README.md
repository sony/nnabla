# Penn Treebank Examples

---

## Overview

This example script implements training of language model on Penn Treebank (PTB) dataset using LSTM. 
It is trained to predict the next word given a sequence of previous words.
The PTB dataset will be automatically downloaded by running the script.

---

## Training (`train.py`)


Run PTB training by

```
python train.py
```

Training perplexity will be displayed throughout the training process, and validation perplexity will be 
displayed at the end of every epoch (1 epoch is roughly about 1300 iterations).
After training, perplexity on test split will be reported, which should be around 82~83. 

By default, the script will be executed with CUDA GPU.
If you prefer to run with CPU, try

```
python train.py -c cpu
```

