# Img2Mol-miniCDDD-ncnn

## Introduction

In this project, I hope to build up a model that read in chemical structural
formula from images and predict the properties, including LogP, molecular
refractivity, Blaban J, number of H acceptors, number of H donors, number of
valence electrons, and topological polar surface area.

- [Img2Mol](https://github.com/bayer-science-for-a-better-life/Img2Mol) is an
accurate SMILES recognition from molecular graphical depictions.

- [CDDD](https://https://github.com/jrwnter/cddd) is the Continuous and
Data-Driven Descriptors for short.

- [miniCDDD](https://github.com/lianghsun/miniCDDD) is an unofficial mini version
of the original Continuous and Data-Driven Descriptors.

## Model Convert

### Img2Mol

In order to convert Img2Mol model, the original model should be download first.

> You can download the trained parameters for the default model (~2.4GB) as
> described in our paper using the following link:
> https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view .
> Please move the downloaded file model.ckpt into the model/ directory.

See Img2Mol readme file for more information.

The Img2Mol model has 2 parts, part 1 encodes an image containing molecular
structures into an embedded array, then part 2 decodes the array and gives
a SMILES string. The encoder is provided by Img2Mol, while the decoder is taken
from another model [CDDD](https://github.com/jrwnter/cddd).

In folder `img2mol_convert`, a jupyter notebook is used to convert the encoder
from PyTorch model to ncnn model via pnnx, and the accuracy has been validated
by calculating the cosine similarity.

### miniCDDD

The next part is to convert CDDD classification model, while the original model
was written in Tensorflow 1.0. Luckily, there is another implemetation with
Keras. So I use a modified version of
[keras2ncnn](https://github.com/MarsTechHAN/keras2ncnn) which supports
1-dimension input, named
[keras2ncnn-1d](https://github.com/mizu-bai/keras2ncnn-1d).

In folder `miniCDDD_convert`, a jupyter notebook is used to convert the
classifier from Keras model to ncnn model. The accuracy has been validated by
calculating the cosine similiarity, same as the Img2Mol model.

## Inference

### Python

A Python based deployment can be found under folder `py_infer`. If you want to
try it, create a folder `model` in it, and copy all the `param` and `bin` files
into it. Then run the following command, all properties predicted by the
Img2Mol + miniCDDD model will be printed.

```bash
$ python3 -u Img2Mol_miniCDDD.py -i ../examples/sample_0.png
Local CDDD installation has not been found.
Initializing Img2Mol Model with random weights.
Setting to `self.eval()`-mode.
Sending model to `cpu` device.
Succesfully created Img2Mol Inference class.
LogP                              2.852758
Molecular Refractivity            81.5513
Balaban J                         1.969035
Number of H Acceptors             2
Number of H Donors                1
Number of Valence Electrons       122
Topological Polar Surface Area    49.32
```

Currently, I still reuse the image preprocessing part in Img2Mol model. It will
be replaced by OpenCV in later C++ deployment.
