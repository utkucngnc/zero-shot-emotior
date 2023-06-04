"""
Python code for detecting multiple faces in an image using the Haar cascade classifier from OpenCV.
"""

import cv2

class MultiFaceDetector:
    def __init__(self, image):
        """
        Initializes a MultiFaceDetector object with an image for face detection.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.faces = self.detect_faces(image)

    def detect_faces(self, image):
        """
        Detects faces in the given image using the Haar cascade classifier.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: A list of bounding boxes representing the detected faces.
        """
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 5, minSize=(40, 40))

        return faces

"""
Module Dependencies:
- cv2: OpenCV, a popular computer vision library for image and video processing.

Code Explanation:
- The code defines a MultiFaceDetector class responsible for detecting multiple faces in an image using the Haar cascade classifier.
- The class takes an image as input during initialization.
- The Haar cascade classifier is loaded using the cv2.CascadeClassifier class, which loads the pre-trained classifier XML file for frontal face detection.
- The detect_faces method takes an image as input, converts it to grayscale, and uses the face_cascade to detect faces in the image.
- The method returns a list of bounding boxes representing the detected faces.

Example Usage:
- Create a MultiFaceDetector object by providing an image for face detection.
- Access the detected faces using the faces attribute of the MultiFaceDetector object.
- The faces attribute contains a list of bounding boxes in the format (x, y, w, h) representing the top-left corner coordinates, width, and height of each detected face.

Note:
- The code assumes that OpenCV (cv2) is properly installed and imported before using the MultiFaceDetector class.
- The Haar cascade classifier XML file ("haarcascade_frontalface_default.xml") is required and should be located in the same directory as the script.

"""