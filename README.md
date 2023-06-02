# Zero-Shot Emotion Detection

This repository contains scripts for performing zero-shot emotion detection on faces in images and real-time video streams using the CLIP model and OpenCV.

## Requirements

- Python 3.6+
- Dependencies (install using `pip install -r requirements.txt`):
  - torch
  - transformers
  - opencv-python

## Scripts

1. **`zero_shot_emotior.py`**: Implements the `ZeroShotEmotior` class for zero-shot emotion detection on faces in an image.
2. **`face_detector.py`**: Defines the `MultiFaceDetector` class for face detection using OpenCV's Haar cascade classifier.
3. **`face_drawer.py`**: Implements the `MultiFaceDrawer` class for drawing bounding boxes and labels around faces in an image.
4. **`config.py`**: Contains model checkpoints, processor, and candidate labels for emotion prediction.
5. **`main.py`**: Provides a sample usage of the `ZeroShotEmotior` class for real-time emotion detection on a video stream.

## Usage

1. Run `python main.py` to start the real-time emotion detection application. Ensure that a compatible camera is connected to your system.
2. Modify the scripts as needed to customize the behavior, integrate with your own projects, or experiment with different models and techniques.

## Contributing

Contributions to the repository are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).