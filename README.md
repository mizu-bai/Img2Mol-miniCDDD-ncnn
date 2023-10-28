# Img2Mol-ncnn

## Introduction

[Img2Mol](https://github.com/bayer-science-for-a-better-life/Img2Mol) is an
accurate SMILES recognition from molecular graphical depictions.

## Model Convert

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
