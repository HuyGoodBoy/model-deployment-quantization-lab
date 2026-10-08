"""Tai pretrained model, chuan bi input dung chung cho ba runtime."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import ARTIFACTS, DATA, INPUT_SHAPE, ROOT, tensorflow, write_json

SAMPLES = {
    "grace_hopper.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
    "sunflower.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
}


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(img, dtype=np.float32)
    # MobileNetV2 preprocess: [0,255] -> [-1,1]. NHWC, batch=1.
    return np.ascontiguousarray((pixels / 127.5 - 1.0)[None, ...])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit phai >= 1")
    tf = tensorflow()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (DATA / "inputs").mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS / "mobilenetv2.keras"
    if not model_path.exists():
        weights_name = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
        weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/" + weights_name
        # cache_dir tuong minh: Keras 2 get_file khong luon dung KERAS_HOME.
        weights_path = tf.keras.utils.get_file(weights_name, weights_url,
                                               cache_dir=str(ROOT / ".cache" / "keras"),
                                               cache_subdir="models")
        model = tf.keras.applications.MobileNetV2(
            weights=weights_path, input_shape=INPUT_SHAPE[1:], alpha=1.0,
            include_top=True, classifier_activation="softmax")
        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
    class_index_path = DATA / "imagenet_class_index.json"
    if not class_index_path.exists():
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            class_index_path.write_bytes(response.read())
    if args.images_dir:
        folder = args.images_dir.resolve()
        if not folder.is_dir():
            raise ValueError(f"Khong co thu muc anh: {folder}")
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        dataset_kind = "user_images"
    else:
        folder = DATA / "images"
        folder.mkdir(parents=True, exist_ok=True)
        for name, url in SAMPLES.items():
            target = folder / name
            if not target.exists():
                print(f"Download sample: {name}", flush=True)
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
        paths = [folder / name for name in SAMPLES]
        dataset_kind = "two_demo_images_not_representative"
    paths = paths[:args.limit]
    if not paths:
        raise ValueError("Khong tim thay anh de danh gia")
    records = []
    for i, path in enumerate(paths):
        x = preprocess(path)
        tensor_path = DATA / "inputs" / f"{i:04d}.npy"
        np.save(tensor_path, x, allow_pickle=False)
        records.append({"filename": path.name, "source_path": str(path.resolve()),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "input_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
                        "input_path": str(tensor_path.relative_to(ROOT))})
    write_json(DATA / "manifest.json", {"dataset_kind": dataset_kind, "records": records,
               "preprocess": "EXIF orientation -> RGB -> resize bilinear 224x224 -> x/127.5-1",
               "input_shape": INPUT_SHAPE, "dtype": "float32"})
    write_json(ARTIFACTS / "model_info.json", {
        "model": "MobileNetV2", "weights": "ImageNet pretrained", "alpha": 1.0,
        "parameters": model.count_params(), "input_shape": INPUT_SHAPE,
        "output_shape": [1, 1000], "output_kind": "softmax probabilities",
        "original_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    })
    print(f"Prepared {len(records)} images and original model: {model_path}", flush=True)


if __name__ == "__main__":
    main()
