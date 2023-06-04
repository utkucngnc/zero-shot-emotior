"""
Python code for drawing bounding boxes and text labels around annotated faces in an image using OpenCV.
"""

import cv2

class MultiFaceDrawer:
    def __init__(self, annotated_faces):
        """
        Initializes a MultiFaceDrawer object with a list of annotated faces.

        Args:
            annotated_faces (list): A list of dictionaries containing face coordinates and corresponding text labels.
        """
        self.annotated_faces = annotated_faces

    def draw_faces(self, image):
        """
        Draws bounding boxes and text labels around annotated faces in the given image.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: The image with bounding boxes and text labels drawn around the annotated faces.
        """
        for data in self.annotated_faces:
            (x, y, w, h) = data['image']
            text = data['text']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image

"""
Module Dependencies:
- cv2: OpenCV, a popular computer vision library for image and video processing.

Code Explanation:
- The code defines a MultiFaceDrawer class responsible for drawing bounding boxes and text labels around annotated faces.
- The class takes a list of annotated faces as input during initialization.
- The draw_faces method takes an image as input and iterates over the annotated faces to draw bounding boxes and text labels around them using OpenCV functions.
- The image with the drawn annotations is then returned.

Example Usage:
- Create a MultiFaceDrawer object by providing a list of annotated faces.
- Call the draw_faces method with an image as input to obtain the image with bounding boxes and text labels drawn around the faces.
- The annotated_faces list should contain dictionaries with 'image' keys representing the face coordinates and 'text' keys representing the corresponding labels.

Note:
- The code assumes that OpenCV (cv2) is properly installed and imported before using the MultiFaceDrawer class.
- The font, color, and line thickness for the bounding boxes and text labels can be customized based on specific requirements.
"""