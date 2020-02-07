# Data augmentation examples

---

## Overview

This example contains several data augmentation methods based on two-sample interpolation.
Currently, the following methods are implemented.
- Mixup [ICLR 2018]: https://arxiv.org/abs/1710.09412
- Cutmix [ICCV 2019]: http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf
- VH-Mixup [WACV 2019]: https://arxiv.org/abs/1805.11272

---

## Start training

You can start training with mixup by the following command.
```
python classification_MDL.py --mixtype mixup --alpha 1.0
```
The training dataset will be automatically downloaded if needed.
(Once downloaded, it will be cached for the next training)

The hyper-parameter alpha controls the distribution that a ratio for the sample interpolation follows. If non-positive value is set, the augmentation will not be conducted.

If you want to use cutmix or vh-mixup, you can use them by changing the "-mixtype" option.
```
python classification_MDL.py --mixtype cutmix --alpha 1.0
python classification_MDL.py --mixtype vhmixup --alpha 1.0
```

The default setting is to train ResNet18 with CIFAR-10 dataset.

---

## License

See `LICENSE`.
