# Expression AI

Expression AI is a local-first Python MVP for facial emotion recognition using the FER2013 dataset, a ResNet18 classifier, single-image inference, and a Tkinter webcam demo.

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

2. Download the FER2013 CSV from Kaggle and place it at `data/fer2013.csv`.

3. Train the model:

   ```bash
   python train/train.py --data data/fer2013.csv --epochs 5
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
- The webcam app uses OpenCV Haar cascades to find the most prominent face before running emotion inference.
