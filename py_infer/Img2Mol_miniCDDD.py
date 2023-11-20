import argparse
import os

import ncnn
import numpy as np
from img2mol.inference import Img2MolInference


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Input image",
        required=True,
    )
    args = parser.parse_args()
    return args


def _img2mol_preprocess(
    img_file: str,
) -> np.array:
    img2mol = Img2MolInference()
    imgs = img2mol.read_image_to_tensor(args.image).clone().detach().numpy()

    return imgs


def _img2mol_infer(
    imgs: np.array,
) -> np.array:
    out = []

    with ncnn.Net() as net:
        net.load_param(os.path.join("model", "img2mol.ncnn.param"))
        net.load_model(os.path.join("model", "img2mol.ncnn.bin"))

        for img in imgs:
            with net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(img))
                _, out0 = ex.extract("out0")
                out.append(np.array(out0))

    out = np.array(out)
    cddd = np.median(out, axis=0)

    return cddd


def _miniCDDD_classifier_infer(
    cddd: np.array,
) -> np.array:
    with ncnn.Net() as net:

        net.load_param(os.path.join("model", "classifier.param"))
        net.load_model(os.path.join("model", "classifier.bin"))

        with net.create_extractor() as ex:
            ex.input("input_2_blob", ncnn.Mat(cddd))
            _, out = ex.extract("classification_output_blob")

    return np.array(out)


def _miniCDDD_postprocess(
    norm_res: np.array,
) -> np.array:
    mean = np.array([
        2.367192268371582, 82.88204193115234, 1.960821509361267,
        2.397263288497925, 1.4187270402908325, 113.73945617675781,
        49.36085510253906,
    ])

    std = np.array([
        1.4581297636032104, 15.581748962402344, 0.4299713373184204,
        0.7589849829673767, 0.8719147443771362, 20.4494686126709,
        17.573997497558594,
    ])

    return norm_res * std + mean


def _print_summary(
    res: np.array,
) -> None:
    print(f"LogP                              {res[0]:.6f}")
    print(f"Molecular Refractivity            {res[1]:.4f}")
    print(f"Balaban J                         {res[2]:.6f}")
    print(f"Number of H Acceptors             {int(res[3])}")
    print(f"Number of H Donors                {int(res[4])}")
    print(f"Number of Valence Electrons       {int(res[5])}")
    print(f"Topological Polar Surface Area    {res[6]:.2f}")


if __name__ == "__main__":
    # parse args
    args = _parse_args()

    # preprocessing
    imgs = _img2mol_preprocess(args.image)

    # Img2Mol
    cddd = _img2mol_infer(imgs)

    # miniCDDD
    norm_res = _miniCDDD_classifier_infer(cddd)

    # postprocess
    res = _miniCDDD_postprocess(norm_res)

    # print summary
    _print_summary(res)
