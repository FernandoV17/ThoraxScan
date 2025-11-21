import threading
from typing import Any, Dict, List

import cv2
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
    def normalize_image(self, image: Image.Image, method: str = "clahe") -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                img_array = np.array(image)

                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                if method == "clahe":
                    normalized = self._normalize_clahe(img_array)
                elif method == "histogram":
                    normalized = self._normalize_histogram(img_array)
                elif method == "global":
                    normalized = self._normalize_global(img_array)
                elif method == "contrast_stretch":
                    normalized = self._normalize_contrast_stretch(img_array)
                elif method == "local":
                    normalized = self._normalize_local(img_array)
                elif method == "percentile":
                    normalized = self._normalize_percentile(img_array)
                else:
                    normalized = self._normalize_clahe(img_array)

                logger.debug(f"Imagen normalizada con OpenCV: {method}")
                return Image.fromarray(normalized)

            except Exception as e:
                logger.error(f"Error normalizando imagen con OpenCV: {e}")
                return image

    def _normalize_clahe(self, img_array: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img_array)

    def _normalize_histogram(self, img_array: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(img_array)

    def _normalize_global(self, img_array: np.ndarray) -> np.ndarray:
        normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _normalize_contrast_stretch(self, img_array: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 > p2:
            stretched = np.clip((img_array - p2) * (255.0 / (p98 - p2)), 0, 255)
            return stretched.astype(np.uint8)
        return img_array

    def _normalize_local(self, img_array: np.ndarray) -> np.ndarray:
        img_float = img_array.astype(np.float32)

        local_mean = cv2.GaussianBlur(img_float, (15, 15), 0)
        local_sq_mean = cv2.GaussianBlur(img_float**2, (15, 15), 0)
        local_std = np.sqrt(local_sq_mean - local_mean**2)

        local_std[local_std < 1] = 1

        normalized = (img_float - local_mean) / local_std
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _normalize_percentile(self, img_array: np.ndarray) -> np.ndarray:
        p5, p95 = np.percentile(img_array, (5, 95))
        if p95 > p5:
            normalized = np.clip((img_array - p5) * (255.0 / (p95 - p5)), 0, 255)
            return normalized.astype(np.uint8)
        return img_array

    @log_function_call()
    def segment_image(
        self, image: Image.Image, method: str = "otsu", **kwargs
    ) -> Image.Image:
        with self._lock:
            if not image:
                return image

            try:
                img_array = np.array(image)

                # If kwargs do not provide params, allow reading from attributes set by UI helpers
                params = dict(kwargs)
                if not params:
                    # adaptive params
                    params = getattr(self, "adaptive_params", {}) or params
                    # kmeans param
                    if not params and hasattr(self, "kmeans_k"):
                        params = {"k": getattr(self, "kmeans_k")}

                if method == "otsu":
                    return self._segment_otsu(img_array)
                elif method == "adaptive":
                    block_size = params.get("block_size", 11)
                    return self._segment_adaptive(img_array, block_size)
                elif method == "threshold":
                    threshold = params.get("threshold", 128)
                    return self._segment_threshold(img_array, threshold)
                elif method == "kmeans":
                    k = params.get("k", 3)
                    return self._segment_kmeans(img_array, k=k)
                elif method in ("erode", "dilate"):
                    kernel = params.get("kernel", 3)
                    iterations = params.get("iterations", 1)
                    if method == "erode":
                        return self._morph_erode(img_array, kernel_size=kernel, iterations=iterations)
                    else:
                        return self._morph_dilate(img_array, kernel_size=kernel, iterations=iterations)
                else:
                    logger.warning(f"Método de segmentación no reconocido: {method}")
                    return image

            except Exception as e:
                logger.error(f"Error en segmentación {method}: {e}")
                return image

    def _segment_otsu(self, img_array: np.ndarray) -> Image.Image:
        try:
            import cv2

            # Ensure grayscale
            if img_array.ndim == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array

            _, thresh = cv2.threshold(
                img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return Image.fromarray(thresh)
        except ImportError:
            # Fallback manual
            from PIL import ImageOps
            if img_array.ndim == 3:
                img_gray = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                return ImageOps.autocontrast(img_gray)
            return ImageOps.autocontrast(Image.fromarray(img_array))

    def _segment_adaptive(
        self, img_array: np.ndarray, block_size: int = 11
    ) -> Image.Image:
        try:
            import cv2

            # Ensure odd block size
            block_size = int(block_size)
            if block_size < 3:
                block_size = 3
            block_size = block_size if block_size % 2 == 1 else block_size + 1

            # Convert to grayscale if needed
            if img_array.ndim == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array

            thresh = cv2.adaptiveThreshold(
                img_gray,
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
        # Convert to grayscale if needed
        try:
            import cv2

            if img_array.ndim == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array

            thresh = (img_gray > int(threshold)).astype(np.uint8) * 255
            return Image.fromarray(thresh)
        except Exception:
            thresh = (img_array > int(threshold)).astype(np.uint8) * 255
            return Image.fromarray(thresh)

    def _segment_kmeans(self, img_array: np.ndarray, k: int = 3, max_iters: int = 30) -> Image.Image:
        try:
            # Work on color or grayscale
            # Determine dimensions
            if img_array.ndim == 2:
                h, w = img_array.shape
                channels = 1
            else:
                h, w, channels = img_array.shape

            # Flatten pixels to (n_pixels, channels)
            if channels == 1:
                flat = img_array.reshape(-1, 1).astype(np.float32)
            else:
                flat = img_array.reshape(-1, channels).astype(np.float32)

            n_pixels = flat.shape[0]

            # Clamp k to number of pixels
            k = max(1, int(min(k, n_pixels)))

            # Initialize centroids randomly from pixels
            rng = np.random.default_rng(0)
            indices = rng.choice(n_pixels, size=k, replace=False)
            centroids = flat[indices].astype(np.float32)

            for _ in range(max_iters):
                distances = np.sqrt(((flat[:, None] - centroids[None, :]) ** 2).sum(axis=2))
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array(
                    [
                        flat[labels == i].mean(axis=0)
                        if np.any(labels == i)
                        else centroids[i]
                        for i in range(k)
                    ],
                    dtype=np.float32,
                )
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            segmented_flat = centroids[labels].astype(np.uint8)

            # Reshape back to image shape
            if channels == 1:
                segmented = segmented_flat.reshape(h, w)
            else:
                segmented = segmented_flat.reshape(h, w, channels)

            return Image.fromarray(segmented)
        except Exception as e:
            logger.error(f"Error en k-means segmentation: {e}")
            return Image.fromarray(img_array)

    def _morph_erode(self, img_array: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> Image.Image:
        try:
            import cv2

            if kernel_size <= 0:
                kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            if img_array.ndim == 3:
                # Work on grayscale conversion for morphology
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            eroded = cv2.erode(gray, kernel, iterations=iterations)
            return Image.fromarray(eroded)
        except Exception as e:
            logger.error(f"Error en erosión morfológica: {e}")
            return Image.fromarray(img_array)

    def _morph_dilate(self, img_array: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> Image.Image:
        try:
            import cv2

            if kernel_size <= 0:
                kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            if img_array.ndim == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            dilated = cv2.dilate(gray, kernel, iterations=iterations)
            return Image.fromarray(dilated)
        except Exception as e:
            logger.error(f"Error en dilatación morfológica: {e}")
            return Image.fromarray(img_array)

    @log_function_call()
    def compute_mask_stats(self, mask_image) -> dict:
        """Compute area and perimeter from a mask or labeled image.

        Returns a dict with keys for each label (as str) and values {'area':int,'perimeter':float}.
        If mask_image is a PIL Image, it will be converted to np.ndarray.
        """
        try:
            if hasattr(mask_image, "convert"):
                # PIL Image
                arr = np.array(mask_image)
            else:
                arr = np.array(mask_image)

            import cv2

            # If RGB, collapse to single channel by taking first channel
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            # Normalize mask values to 0..255 uint8
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            stats = {}

            unique_vals = np.unique(arr)
            for val in unique_vals:
                # treat background value as 0 but still compute if user wants
                mask = (arr == val).astype(np.uint8) * 255
                # find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area = int(np.sum(mask > 0))
                perimeter = 0.0
                for c in contours:
                    perimeter += float(cv2.arcLength(c, True))

                stats[str(int(val))] = {"area": area, "perimeter": perimeter}

            return stats
        except Exception as e:
            logger.error(f"Error computing mask stats: {e}")
            return {}

    @log_function_call()
    def kmeans_cluster_stats(self, image: Image.Image, k: int = 3, max_iters: int = 30) -> list:
        """Run k-means and return per-cluster area/perimeter stats.

        Returns a list of dicts: [{'label':i,'area':..., 'perimeter':..., 'count':...}, ...]
        """
        try:
            img_array = np.array(image)
            # flatten as in _segment_kmeans
            if img_array.ndim == 2:
                h, w = img_array.shape
                flat = img_array.reshape(-1, 1).astype(np.float32)
                channels = 1
            else:
                h, w, channels = img_array.shape
                flat = img_array.reshape(-1, channels).astype(np.float32)

            n_pixels = flat.shape[0]
            k = max(1, int(min(k, n_pixels)))

            rng = np.random.default_rng(0)
            indices = rng.choice(n_pixels, size=k, replace=False)
            centroids = flat[indices].astype(np.float32)

            labels = np.zeros(n_pixels, dtype=np.int32)
            for _ in range(max_iters):
                # compute distances
                distances = np.sqrt(((flat[:, None] - centroids[None, :]) ** 2).sum(axis=2))
                new_labels = np.argmin(distances, axis=1)
                if np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                new_centroids = np.array(
                    [
                        flat[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                        for i in range(k)
                    ],
                    dtype=np.float32,
                )
                centroids = new_centroids

            label_map = labels.reshape(h, w)

            # compute stats per label
            import cv2

            results = []
            for i in range(k):
                mask = (label_map == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area = int(np.sum(mask > 0))
                perimeter = 0.0
                for c in contours:
                    perimeter += float(cv2.arcLength(c, True))

                results.append({"label": i, "area": area, "perimeter": perimeter, "count": int(np.sum(label_map == i))})

            return results
        except Exception as e:
            logger.error(f"Error computing kmeans cluster stats: {e}")
            return []

    @log_function_call()
    def calculate_histogram(self, image: Image.Image) -> Dict[str, Any]:
        with self._lock:
            if not image:
                return {}

            try:
                # Usar cache para mejor rendimiento
                cache_key = f"{id(image)}_{image.size}_{image.mode}"
                if cache_key in self._histogram_cache:
                    return self._histogram_cache[cache_key]

                img_array = np.array(image)
                histogram_data = {}

                if len(img_array.shape) == 3:  # RGB
                    colors = ["red", "green", "blue"]
                    for i, color in enumerate(colors):
                        channel_data = img_array[:, :, i]
                        hist, bins = np.histogram(
                            channel_data, bins=256, range=(0, 255)
                        )
                        histogram_data[color] = {
                            "values": hist.tolist(),
                            "bins": bins[:-1].tolist(),
                        }
                    histogram_data["type"] = "rgb"

                else:  # Escala de grises
                    hist, bins = np.histogram(img_array, bins=256, range=(0, 255))
                    histogram_data["gray"] = {
                        "values": hist.tolist(),
                        "bins": bins[:-1].tolist(),
                    }
                    histogram_data["type"] = "grayscale"

                flat_array = img_array.flatten()
                stats = {
                    "mean": float(np.mean(flat_array)),
                    "std": float(np.std(flat_array)),
                    "min": int(np.min(flat_array)),
                    "max": int(np.max(flat_array)),
                    "median": int(np.median(flat_array)),
                    "total_pixels": int(flat_array.size),
                    "dynamic_range": int(np.max(flat_array) - np.min(flat_array)),
                }

                try:
                    if histogram_data["type"] == "grayscale":
                        mode_index = np.argmax(histogram_data["gray"]["values"])
                        stats["mode"] = int(mode_index)
                    else:
                        modes = []
                        for color in ["red", "green", "blue"]:
                            mode_index = np.argmax(histogram_data[color]["values"])
                            modes.append(int(mode_index))
                        stats["mode"] = int(np.mean(modes))
                except:
                    stats["mode"] = 0

                histogram_data["stats"] = stats

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
