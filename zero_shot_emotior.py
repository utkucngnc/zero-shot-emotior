"""
Python code for zero-shot emotion detection on faces in an image using the CLIP model and OpenCV.
"""

import torch
from face_detector import MultiFaceDetector
from face_drawer import MultiFaceDrawer
from config import model, processor, candidate_labels
from typing import List, Dict, Any, Tuple, Union

class ZeroShotEmotior:
    def __init__(self, image: Union[str, Any]):
        """
        Initializes a ZeroShotEmotior object for zero-shot emotion detection on faces in an image.

        Args:
            image (Union[str, Any]): The input image as a file path or image data.

        Attributes:
            annotated_faces_list (List[Dict[str, Union[Tuple[int, int, int, int], str]]]): A list of dictionaries representing annotated faces,
                where each dictionary contains the face coordinates and the predicted emotion label.
            frame (Any): The image with annotated faces and emotion labels drawn.

        """
        self.image = image
        self.annotated_faces_list: List[Dict[str, Union[Tuple[int, int, int, int], str]]] = []
        self.frame = self.draw_on_feed()
        
    def draw_on_feed(self) -> Any:
        """
        Draws bounding boxes and emotion labels around faces in the input image.

        Returns:
            Any: The image with annotated faces and emotion labels drawn.

        """
        self.crop_and_forward()
        return MultiFaceDrawer(self.annotated_faces_list).draw_faces(self.image)
        
    def crop_and_forward(self) -> None:
        """
        Performs face detection, emotion prediction, and annotation for each face in the input image.

        """
        faces = MultiFaceDetector(self.image).faces
        for (x, y, w, h) in faces:
            cropped_image = self.image[y:y+h, x:x+w]
            inputs = processor(images=cropped_image, text=candidate_labels, return_tensors="pt", padding=True)
            result = self.predict(inputs)
            image_text = result[0]['label']
            self.annotated_faces_list.append({'image': (x, y, w, h), 'text': image_text})
            
    def predict(self, inputs: Dict[str, Any]) -> List[Dict[str, Union[float, str]]]:
        """
        Performs emotion prediction on the cropped face image using the CLIP model.

        Args:
            inputs (Dict[str, Any]): The input data for the emotion prediction.

        Returns:
            List[Dict[str, Union[float, str]]]: A list of dictionaries representing the predicted emotion scores and labels for the face.

        """
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).numpy()
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
        ]
        
        return result

"""
Module Dependencies:
- torch: The PyTorch library for deep learning.

Code Explanation:
- The code defines a ZeroShotEmotior class responsible for performing zero-shot emotion detection on faces in an image.
- The class takes an image as input during initialization.
- The annotated_faces_list attribute is a list that will store dictionaries representing annotated faces and their predicted emotion labels.
- The frame attribute stores the image with annotated faces and emotion labels drawn.
- The draw_on_feed method draws bounding boxes and emotion labels around the faces in the input image using the MultiFaceDrawer class.
- The crop_and_forward method performs face detection, emotion prediction, and annotation for each face in the input image.
- The predict method performs emotion prediction on the cropped face image using the CLIP model.
- The ZeroShotEmotior class relies on the face detection functionality provided by the MultiFaceDetector class.

Example Usage:
- Create a ZeroShotEmotior object by providing an image for zero-shot emotion detection.
- Access the annotated_faces_list attribute to retrieve the annotated faces and their predicted emotion labels.
- Access the frame attribute to obtain the image with annotated faces and emotion labels drawn.

Note:
- The code assumes that the required dependencies (torch, face_detector, face_drawer, config) are properly installed and imported before using the ZeroShotEmotior class.
- The candidate_labels should be a list of emotion labels compatible with the CLIP model.
- The face detection, emotion prediction, and annotation logic can be customized or extended based on specific requirements.
"""