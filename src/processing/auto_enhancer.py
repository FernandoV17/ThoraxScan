import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.helpers.logger import get_module_logger, log_function_call

logger = get_module_logger(__name__)


class AutoEnhancer:
    def __init__(self):
        self.use_gpu = False  # Por defecto sin GPU
        self._lock = threading.RLock()
        self._available_methods = ["auto", "contrast", "sharpness", "denoise", "basic"]

        self.has_opencv = self._check_opencv_availability()

        logger.info(f"AutoEnhancer inicializado - OpenCV: {self.has_opencv}")

    def _check_opencv_availability(self) -> bool:
        try:
            import cv2

            test_array = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0)
            clahe.apply(test_array)
            logger.info("OpenCV disponible para CLAHE")
            return True
        except ImportError:
            logger.info("OpenCV no disponible - usando métodos PIL")
            return False
        except Exception as e:
            logger.warning(f"OpenCV disponible pero con errores: {e}")
            return False

    @log_function_call(log_args=False)
    def enhance(self, image: Image.Image, method: str = "auto") -> Image.Image:
        start_time = time.time()

        with self._lock:
            if not image:
                logger.warning("No hay imagen para mejorar")
                return image

            try:
                enhanced = image.copy()

                if method == "auto":
                    enhanced = self._auto_enhance_pipeline(enhanced)
                elif method == "contrast":
                    enhanced = self._enhance_contrast(enhanced)
                elif method == "sharpness":
                    enhanced = self._enhance_sharpness(enhanced)
                elif method == "denoise":
                    enhanced = self._denoise_image(enhanced)
                elif method == "basic":
                    enhanced = self._basic_enhancement(enhanced)
                else:
                    logger.warning(f"Método no reconocido: {method}, usando auto")
                    enhanced = self._auto_enhance_pipeline(enhanced)

                processing_time = time.time() - start_time
                logger.info(
                    f"Auto-enhance completado ({method}): {processing_time:.3f}s"
                )

                return enhanced

            except Exception as e:
                logger.error(f"Error en auto-enhance: {e}")
                return image

    def _auto_enhance_pipeline(self, image: Image.Image) -> Image.Image:
        image = self._enhance_contrast(image)

        image = self._denoise_image(image)

        image = self._enhance_sharpness(image)

        return image

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        try:
            if self.has_opencv:
                return self._enhance_contrast_clahe(image)
            else:
                return self._enhance_contrast_pil(image)
        except Exception as e:
            logger.warning(f"Mejora de contraste falló, usando básico: {e}")
            return self._basic_contrast_enhancement(image)

    def _enhance_contrast_clahe(self, image: Image.Image) -> Image.Image:
        try:
            import cv2

            img_array = np.array(image)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_array = clahe.apply(img_array)

            return Image.fromarray(enhanced_array)

        except Exception as e:
            logger.warning(f"CLAHE falló: {e}")
            return self._enhance_contrast_pil(image)

    def _enhance_contrast_pil(self, image: Image.Image) -> Image.Image:
        try:
            equalized = ImageOps.equalize(image)

            enhancer = ImageEnhance.Contrast(equalized)
            return enhancer.enhance(1.2)

        except Exception as e:
            logger.warning(f"PIL contrast falló: {e}")
            return image

    def _basic_contrast_enhancement(self, image: Image.Image) -> Image.Image:
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.3)
        except:
            return image

    def _denoise_image(self, image: Image.Image) -> Image.Image:
        try:
            return image.filter(ImageFilter.MedianFilter(size=3))
        except Exception as e:
            logger.warning(f"Denoise falló: {e}")
            return image

    def _enhance_sharpness(self, image: Image.Image) -> Image.Image:
        try:
            return image.filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2)
            )
        except Exception as e:
            logger.warning(f"Sharpness enhancement falló: {e}")
            return image

    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        except:
            return image

    def get_available_methods(self) -> list:
        return self._available_methods.copy()

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "opencv_available": self.has_opencv,
            "gpu_available": self.use_gpu,
            "methods": self._available_methods,
        }
