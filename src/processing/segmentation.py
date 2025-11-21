import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class AdvancedSegmentation:
    def __init__(self):
        self.logger = get_module_logger(__name__)

    def threshold_segmentation(self, image, method="otsu", threshold_value=127):
        """Segmentación por umbral"""
        img_array = np.array(image.convert("L"))

        if method == "otsu":
            _, segmented = cv2.threshold(
                img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            segmented = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "manual":
            _, segmented = cv2.threshold(
                img_array, threshold_value, 255, cv2.THRESH_BINARY
            )

        return Image.fromarray(segmented), self.calculate_properties(segmented)

    def kmeans_segmentation(self, image, k=3):
        """Segmentación por K-means"""
        img_array = np.array(image.convert("RGB"))
        pixels = img_array.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        segmented = labels.reshape(img_array.shape[:2])

        # Convertir a imagen binaria (usar el cluster más oscuro como fondo)
        segmented_binary = (
            segmented != np.argmin(kmeans.cluster_centers_.sum(axis=1))
        ).astype(np.uint8) * 255

        return Image.fromarray(segmented_binary), self.calculate_properties(
            segmented_binary
        )

    def morphological_operations(self, image, operation="erode", kernel_size=3):
        """Operaciones morfológicas"""
        img_array = np.array(image.convert("L"))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "erode":
            result = cv2.erode(img_array, kernel, iterations=1)
        elif operation == "dilate":
            result = cv2.dilate(img_array, kernel, iterations=1)
        elif operation == "open":
            result = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(result), self.calculate_properties(result)

    def calculate_properties(self, binary_image):
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        properties = {
            "num_regions": len(contours),
            "total_area": 0,
            "total_perimeter": 0,
            "regions": [],
        }

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            properties["total_area"] += area
            properties["total_perimeter"] += perimeter
            properties["regions"].append({"area": area, "perimeter": perimeter})

        return properties
