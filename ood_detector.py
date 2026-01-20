import mediapipe as mp
import numpy as np

class OODDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # static_image_mode=True for processing single images
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

    def is_ood(self, image):
        """
        Checks if the image contains OOD elements (Hands or Faces).
        Args:
            image: A PIL Image or numpy array (RGB).
        Returns:
            bool: True if OOD detected, False otherwise.
        """
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        # Check for hands
        results_hands = self.hands.process(image_np)
        if results_hands.multi_hand_landmarks:
            return True # Hand detected
            
        # Check for faces
        results_face = self.face_detection.process(image_np)
        if results_face.detections:
            return True # Face detected
            
        return False
