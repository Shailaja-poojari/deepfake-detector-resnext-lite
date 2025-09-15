# Placeholder for MediaPipe-based face detector"""
Face detection wrapper using MediaPipe.
"""
import mediapipe as mp
import cv2
from typing import List

mp_face = mp.solutions.face_detection

def crop_faces_from_image(image) -> List:
    """
    Detect and crop faces from a single image.

    Args:
        image: OpenCV BGR image.

    Returns:
        List of cropped face images.
    """
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        results = fd.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []

        h, w = image.shape[:2]
        crops = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)
            crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            crops.append(crop)
        return crops
