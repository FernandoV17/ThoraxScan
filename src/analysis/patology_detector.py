from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import feature, filters, measure

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class PatologyDetector:
    def __init__(self):
        logger.info("PatologyDetector inicializado")

    def detect_consolidation(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image)
            results = {}

            lbp_features = self._texture_analysis_lbp(img_array)
            results["lbp_analysis"] = lbp_features

            edges = self._edge_detection_canny(img_array)
            results["edge_density"] = np.sum(edges) / edges.size

            regions = self._connected_component_analysis(img_array)
            results["region_analysis"] = regions

            gabor_features = self._gabor_filter_analysis(img_array)
            results["gabor_analysis"] = gabor_features

            local_hist = self._local_histogram_analysis(img_array)
            results["local_histogram"] = local_hist

            confidence = self._combine_detection_results(results)

            return {
                "detected": confidence > 0.6,
                "confidence": confidence,
                "methods_used": list(results.keys()),
                "details": results,
            }

        except Exception as e:
            logger.error(f"Error en detección de consolidación: {e}")
            return {"detected": False, "confidence": 0, "error": str(e)}

    def detect_opacities(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image)
            results = {}

            adaptive_thresh = self._adaptive_thresholding(img_array)
            results["adaptive_threshold"] = adaptive_thresh

            dog_features = self._difference_of_gaussians(img_array)
            results["dog_analysis"] = dog_features

            hough_circles = self._hough_circle_detection(img_array)
            results["hough_circles"] = hough_circles

            asymmetry = self._asymmetry_analysis(img_array)
            results["asymmetry"] = asymmetry

            confidence = self._combine_opacity_results(results)

            return {
                "detected": confidence > 0.5,
                "confidence": confidence,
                "methods_used": list(results.keys()),
                "details": results,
            }

        except Exception as e:
            logger.error(f"Error en detección de opacidades: {e}")
            return {"detected": False, "confidence": 0, "error": str(e)}

    def _texture_analysis_lbp(self, img_array: np.ndarray) -> Dict[str, float]:
        try:
            lbp = feature.local_binary_pattern(img_array, 8, 1, method="uniform")

            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            hist = hist.astype("float")
            hist /= hist.sum() + 1e-7  # Normalizar

            return {
                "lbp_entropy": -np.sum(hist * np.log2(hist + 1e-7)),
                "lbp_energy": np.sum(hist**2),
                "lbp_contrast": np.std(hist),
            }
        except Exception as e:
            logger.warning(f"Error en análisis LBP: {e}")
            return {}

    def _edge_detection_canny(self, img_array: np.ndarray) -> np.ndarray:
        try:
            edges = feature.canny(img_array, sigma=2)
            return edges
        except:
            return np.zeros_like(img_array, dtype=bool)

    def _connected_component_analysis(self, img_array: np.ndarray) -> Dict[str, Any]:
        try:
            thresh = filters.threshold_otsu(img_array)
            binary = img_array > thresh

            # Etiquetar componentes
            labeled_array, num_features = ndimage.label(binary)

            properties = measure.regionprops(labeled_array)

            areas = [prop.area for prop in properties]
            eccentricities = [prop.eccentricity for prop in properties]

            return {
                "num_regions": num_features,
                "avg_area": np.mean(areas) if areas else 0,
                "max_area": np.max(areas) if areas else 0,
                "avg_eccentricity": np.mean(eccentricities) if eccentricities else 0,
            }
        except Exception as e:
            logger.warning(f"Error en análisis de componentes: {e}")
            return {}

    def _gabor_filter_analysis(self, img_array: np.ndarray) -> Dict[str, float]:
        try:
            gabor_real, gabor_imag = filters.gabor(img_array, frequency=0.1)
            gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)

            return {
                "gabor_mean": np.mean(gabor_mag),
                "gabor_std": np.std(gabor_mag),
                "gabor_energy": np.sum(gabor_mag**2),
            }
        except:
            return {}

    def _local_histogram_analysis(self, img_array: np.ndarray) -> Dict[str, float]:
        try:
            h, w = img_array.shape
            block_h, block_w = h // 4, w // 4

            local_means = []
            local_stds = []

            for i in range(4):
                for j in range(4):
                    block = img_array[
                        i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w
                    ]
                    local_means.append(np.mean(block))
                    local_stds.append(np.std(block))

            return {
                "local_contrast": np.std(local_means),
                "local_variability": np.mean(local_stds),
            }
        except:
            return {}

    def _adaptive_thresholding(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Umbralización adaptativa"""
        try:
            thresh = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return {
                "white_pixels": np.sum(thresh == 255) / thresh.size,
                "black_pixels": np.sum(thresh == 0) / thresh.size,
            }
        except:
            return {}

    def _combine_detection_results(self, results: Dict[str, Any]) -> float:
        confidence = 0.0
        weights = {
            "lbp_analysis": 0.3,
            "edge_density": 0.2,
            "region_analysis": 0.25,
            "gabor_analysis": 0.15,
            "local_histogram": 0.1,
        }

        for method, weight in weights.items():
            if method in results and results[method]:
                if method == "lbp_analysis":
                    entropy = results[method].get("lbp_entropy", 0)
                    confidence += weight * min(entropy / 2.0, 1.0)
                elif method == "edge_density":
                    density = results[method]
                    confidence += weight * min(density * 10, 1.0)

        return min(confidence, 1.0)

    def _combine_opacity_results(self, results: Dict[str, Any]) -> float:
        confidence = 0.0
        # Lógica de combinación similar...
        return min(confidence, 1.0)
