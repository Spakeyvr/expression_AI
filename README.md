# Expression AI

Expression AI is a local-first Python MVP for facial emotion recognition using either the FER2013 CSV dataset or split image folders, an EfficientNet-B0 classifier, single-image inference, and a Tkinter webcam demo.

## Project Layout

```text
expression_AI/
├── data/
│   └── dataset.py
├── model/
│   ├── checkpoints/
│   └── model.py
├── train/
│   └── train.py
├── display/
│   ├── app.py
│   └── images/
├── infer.py
├── common.py
├── requirements.txt
└── tests/
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Add a dataset in one of these formats:

   `FER2013` CSV:

   ```text
   data/fer2013.csv
   ```

   Split image folders:

   ```text
   data/archive/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```

   Alternate split image folders with numeric labels are also supported:

   ```text
   data/data/archive(1)/
   ├── DATASET/
   │   ├── train/
   │   │   ├── 1/
   │   │   ├── 2/
   │   │   ├── 3/
   │   │   ├── 4/
   │   │   ├── 5/
   │   │   ├── 6/
   │   │   └── 7/
   │   └── test/
   │       ├── 1/
   │       ├── 2/
   │       ├── 3/
   │       ├── 4/
   │       ├── 5/
   │       ├── 6/
   │       └── 7/
   ├── train_labels.csv
   └── test_labels.csv
   ```

   Numeric class folders map to the project's emotion order:
   `1=Angry`, `2=Disgust`, `3=Fear`, `4=Happy`, `5=Sad`, `6=Surprise`, `7=Neutral`

   `val/` is also supported. If you only have `train/` and `test/`, the trainer will automatically use `test/` as the validation split.

   You can also point training at a parent directory that contains multiple archives, including nested layouts like:

   ```text
   data/data/
   ├── archive/
   │   ├── train/
   │   └── val/
   └── archive_2/
       ├── train/
       └── val/
   ```

   When you pass that parent folder to `--data`, Expression AI will discover each archive that contains the requested split and combine them into one dataset.

3. Train the model:

   ```bash
   python train/train.py --data data/fer2013.csv --epochs 5
   ```

   or

   ```bash
   python train/train.py --data data/archive --epochs 5
   ```

   or train from a parent directory that contains multiple archives:

   ```bash
   python train/train.py --data data/data --epochs 5
   ```

   or train directly from the numeric-label dataset above:

   ```bash
   python train/train.py --data 'data/data/archive(1)' --epochs 5
   ```

4. Run inference on a single image:

   ```bash
   python infer.py --image path/to/image.jpg --checkpoint model/checkpoints/best.pt
   ```

5. Launch the webcam demo:

   ```bash
   python display/app.py --checkpoint model/checkpoints/best.pt
   ```

## Notes

- `--device auto` prefers Apple GPU acceleration through MPS when it is available at runtime and otherwise falls back to CPU.
- `--smoke-run` performs a tiny 1-epoch training pass for local verification.
- `--subset` now samples examples randomly (with `--subset-seed`) to avoid class-order bias during quick runs.
- The model uses torchvision's current EfficientNet-B0 weights API, so the `--pretrained` path follows the same ImageNet weights behavior as the library docs.
- The webcam app uses a multi-angle OpenCV cascade pass (frontal, mirrored profile, and slight tilt recovery) before running emotion inference.
- Training now applies horizontal-flip and rotation augmentation so the classifier is less brittle on off-axis face crops.
