import threading
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.helpers.logger import get_module_logger, log_function_call

logger = get_module_logger(__name__)


class ManualAdjustments:
    def __init__(self):
        self._lock = threading.RLock()
        self._histogram_cache = {}
        logger.info("ManualAdjustments inicializado")

    @log_function_call()
    def adjust_brightness_contrast(
        self, image: Image.Image, brightness: float = 1.0, contrast: float = 1.0
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                enhanced = image.copy()

                if brightness != 1.0:
                    brightness_enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = brightness_enhancer.enhance(brightness)

                if contrast != 1.0:
                    contrast_enhancer = ImageEnhance.Contrast(enhanced)
                    enhanced = contrast_enhancer.enhance(contrast)

                logger.debug(f"Brillo: {brightness:.2f}, Contraste: {contrast:.2f}")
                return enhanced

            except Exception as e:
                logger.error(f"Error ajustando brillo/contraste: {e}")
                return image

    @log_function_call()
    def adjust_levels(
        self,
        image: Image.Image,
        black_point: int = 0,
        white_point: int = 255,
        gamma: float = 1.0,
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                img_array = np.array(image, dtype=np.float32)

                if black_point > 0 or white_point < 255:
                    img_array = np.clip(
                        (img_array - black_point)
                        * (255.0 / (white_point - black_point)),
                        0,
                        255,
                    )

                if gamma != 1.0:
                    img_array = 255.0 * (img_array / 255.0) ** (1.0 / gamma)

                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                result = Image.fromarray(img_array)

                logger.debug(
                    f"Niveles: black={black_point}, white={white_point}, gamma={gamma:.2f}"
                )
                return result

            except Exception as e:
                logger.error(f"Error ajustando niveles: {e}")
                return image

    @log_function_call()
    def adjust_gamma(self, image: Image.Image, gamma: float) -> Image.Image:
        with self._lock:
            if not image or gamma <= 0:
                return image

            try:
                img_array = np.array(image, dtype=np.float32)
                img_array = 255.0 * (img_array / 255.0) ** (1.0 / gamma)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                result = Image.fromarray(img_array)
                logger.debug(f"Gamma ajustado: {gamma:.2f}")
                return result

            except Exception as e:
                logger.error(f"Error ajustando gamma: {e}")
                return image

    @log_function_call()
    def normalize_image(
        self, image: Image.Image, method: str = "global"
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                img_array = np.array(image)

                if method == "global":
                    normalized = (
                        (img_array - img_array.min())
                        / (img_array.max() - img_array.min())
                        * 255
                    )
                    normalized = normalized.astype(np.uint8)

                elif method == "histogram":
                    normalized = ImageOps.equalize(Image.fromarray(img_array))
                    return normalized

                elif method == "adaptive":
                    mean = img_array.mean()
                    std = img_array.std()
                    normalized = (img_array - mean) / std * 64 + 128  # Centrar en 128
                    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

                else:
                    logger.warning(f"Método de normalización no reconocido: {method}")
                    return image

                result = Image.fromarray(normalized)
                logger.debug(f"Imagen normalizada: {method}")
                return result

            except Exception as e:
                logger.error(f"Error normalizando imagen: {e}")
                return image

    @log_function_call()
    def segment_image(
        self, image: Image.Image, method: str = "otsu", **kwargs
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                img_array = np.array(image)

                if method == "otsu":
                    return self._segment_otsu(img_array)
                elif method == "adaptive":
                    block_size = kwargs.get("block_size", 11)
                    return self._segment_adaptive(img_array, block_size)
                elif method == "threshold":
                    threshold = kwargs.get("threshold", 128)
                    return self._segment_threshold(img_array, threshold)
                else:
                    logger.warning(f"Método de segmentación no reconocido: {method}")
                    return image

            except Exception as e:
                logger.error(f"Error en segmentación {method}: {e}")
                return image

    def _segment_otsu(self, img_array: np.ndarray) -> Image.Image:
        try:
            import cv2

            _, thresh = cv2.threshold(
                img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return Image.fromarray(thresh)
        except ImportError:
            # Fallback manual
            from PIL import ImageOps

            return ImageOps.autocontrast(Image.fromarray(img_array))

    def _segment_adaptive(
        self, img_array: np.ndarray, block_size: int = 11
    ) -> Image.Image:
        try:
            import cv2

            block_size = block_size if block_size % 2 == 1 else block_size + 1
            thresh = cv2.adaptiveThreshold(
                img_array,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                2,
            )
            return Image.fromarray(thresh)
        except ImportError:
            logger.warning("OpenCV no disponible para segmentación adaptativa")
            return Image.fromarray(img_array)

    def _segment_threshold(self, img_array: np.ndarray, threshold: int) -> Image.Image:
        thresh = (img_array > threshold).astype(np.uint8) * 255
        return Image.fromarray(thresh)

    @log_function_call()
    def calculate_histogram(self, image: Image.Image) -> Dict[str, Any]:
        with self._lock:
            if not image:
                return {}

            try:
                cache_key = f"{id(image)}_{image.size}_{image.mode}"
                if cache_key in self._histogram_cache:
                    return self._histogram_cache[cache_key]

                img_array = np.array(image)
                histogram_data = {}

                if len(img_array.shape) == 3:  # RGB
                    colors = ["red", "green", "blue"]
                    for i, color in enumerate(colors):
                        hist, bins = np.histogram(
                            img_array[:, :, i], bins=256, range=(0, 255)
                        )
                        histogram_data[color] = {
                            "values": hist.tolist(),
                            "bins": bins[:-1].tolist(),  # Excluir el último bin
                        }
                    histogram_data["type"] = "rgb"

                else:
                    hist, bins = np.histogram(img_array, bins=256, range=(0, 255))
                    histogram_data["gray"] = {
                        "values": hist.tolist(),
                        "bins": bins[:-1].tolist(),
                    }
                    histogram_data["type"] = "grayscale"

                histogram_data["stats"] = {
                    "mean": float(np.mean(img_array)),
                    "std": float(np.std(img_array)),
                    "min": int(np.min(img_array)),
                    "max": int(np.max(img_array)),
                    "median": int(np.median(img_array)),
                }

                self._histogram_cache[cache_key] = histogram_data
                if len(self._histogram_cache) > 10:  # Limitar cache
                    self._histogram_cache.pop(next(iter(self._histogram_cache)))

                return histogram_data

            except Exception as e:
                logger.error(f"Error calculando histograma: {e}")
                return {}

    @log_function_call()
    def apply_filter(
        self, image: Image.Image, filter_type: str, **kwargs
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                filter_map = {
                    "sharpen": ImageFilter.SHARPEN,
                    "blur": ImageFilter.BLUR,
                    "edge_enhance": ImageFilter.EDGE_ENHANCE,
                    "smooth": ImageFilter.SMOOTH,
                    "detail": ImageFilter.DETAIL,
                    "emboss": ImageFilter.EMBOSS,
                    "contour": ImageFilter.CONTOUR,
                    "find_edges": ImageFilter.FIND_EDGES,
                }

                if filter_type in filter_map:
                    result = image.filter(filter_map[filter_type])
                    logger.debug(f"Filtro aplicado: {filter_type}")
                    return result
                elif filter_type == "gaussian_blur":
                    radius = kwargs.get("radius", 2)
                    result = image.filter(ImageFilter.GaussianBlur(radius=radius))
                    logger.debug(f"Gaussian blur aplicado: radius={radius}")
                    return result
                elif filter_type == "unsharp_mask":
                    radius = kwargs.get("radius", 2)
                    percent = kwargs.get("percent", 150)
                    threshold = kwargs.get("threshold", 3)
                    result = image.filter(
                        ImageFilter.UnsharpMask(
                            radius=radius, percent=percent, threshold=threshold
                        )
                    )
                    logger.debug(f"Unsharp mask aplicado")
                    return result
                else:
                    logger.warning(f"Filtro no reconocido: {filter_type}")
                    return image

            except Exception as e:
                logger.error(f"Error aplicando filtro {filter_type}: {e}")
                return image

    def get_available_filters(self) -> List[str]:
        return [
            "sharpen",
            "blur",
            "edge_enhance",
            "smooth",
            "detail",
            "emboss",
            "contour",
            "find_edges",
            "gaussian_blur",
            "unsharp_mask",
        ]

    def get_segmentation_methods(self) -> List[str]:
        methods = ["otsu", "threshold"]
        try:
            import cv2

            methods.extend(["adaptive"])
        except ImportError:
            pass
        return methods

    def get_normalization_methods(self) -> List[str]:
        return ["global", "histogram", "adaptive"]

    @log_function_call()
    def enhance(self, image: Image.Image, method: str = "auto") -> Image.Image:
        return self._basic_enhancement(image)

    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Mejora básica para compatibilidad"""
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        except:
            return image
