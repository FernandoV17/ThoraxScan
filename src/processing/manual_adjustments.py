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

    def on_segmentation(self):
        """Interfaz de segmentación con parámetros configurables"""
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        seg_window = tk.Toplevel(self.root)
        seg_window.title("Segmentación Avanzada")
        seg_window.geometry("400x500")
        seg_window.configure(bg=self.theme["panel_bg"])

        main_frame = tk.Frame(seg_window, bg=self.theme["panel_bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        tk.Label(
            main_frame,
            text="Selecciona método de segmentación:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 11, "bold"),
        ).pack(anchor="w", pady=(0, 15))

        # Métodos de segmentación
        methods = [
            ("Umbralización Otsu (Automático)", "otsu"),
            ("Umbral Manual", "threshold"),
            ("Umbralización Adaptativa", "adaptive"),
            ("K-Means Clustering", "kmeans"),
            ("Operaciones Morfológicas", "morphological"),
            ("Watershed", "watershed"),
        ]

        selected_method = tk.StringVar(value="otsu")

        for name, method in methods:
            tk.Radiobutton(
                main_frame,
                text=name,
                variable=selected_method,
                value=method,
                bg=self.theme["panel_bg"],
                fg=self.theme["text_color"],
                selectcolor=self.theme["accent"],
            ).pack(anchor="w", pady=2)

        # Frame para parámetros
        params_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        params_frame.pack(fill="x", pady=15)

        self.segmentation_params = {}

        def update_params(*args):
            # Limpiar frame de parámetros
            for widget in params_frame.winfo_children():
                widget.destroy()

            method = selected_method.get()

            if method == "threshold":
                self._create_threshold_params(params_frame)
            elif method == "adaptive":
                self._create_adaptive_params(params_frame)
            elif method == "kmeans":
                self._create_kmeans_params(params_frame)
            elif method == "morphological":
                self._create_morphological_params(params_frame)

        selected_method.trace("w", update_params)
        update_params()  # Llamar inicialmente

        # Botones de acción
        button_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        button_frame.pack(fill="x", pady=20)

        def apply_segmentation():
            method = selected_method.get()
            params = self.segmentation_params.copy()

            try:
                current_image = self.image_controller.get_current_image()
                if method in ["kmeans", "morphological", "watershed"]:
                    # Usar segmentación avanzada
                    segmented, properties = (
                        self.manual_adjustments.advanced_segmentation(
                            current_image, method, **params
                        )
                    )

                    # Mostrar propiedades
                    props_text = f"Segmentación {method}:\n"
                    props_text += f"Regiones: {properties.get('num_regions', 0)}\n"
                    props_text += (
                        f"Área total: {properties.get('total_area', 0):.0f} px\n"
                    )
                    props_text += f"Perímetro total: {properties.get('total_perimeter', 0):.0f} px"

                    self.show_message("Propiedades Segmentación", props_text)

                else:
                    # Segmentación básica
                    segmented = self.manual_adjustments.segment_image(
                        current_image, method, **params
                    )
                    properties = {}

                self.image_controller.image_manager.update_image(
                    segmented, f"Segmentación: {method}"
                )
                self.display_image()
                self.update_histogram()
                seg_window.destroy()

            except Exception as e:
                logger.error(f"Error en segmentación: {e}")
                self.show_message("Error", f"Error en segmentación: {str(e)}")

        tk.Button(
            button_frame,
            text="Aplicar Segmentación",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
            command=apply_segmentation,
        ).pack(side="left", padx=5)

        tk.Button(
            button_frame,
            text="Cancelar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=seg_window.destroy,
        ).pack(side="right", padx=5)

    def _create_threshold_params(self, parent):
        """Crea controles para parámetros de umbral manual"""
        tk.Label(
            parent,
            text="Umbral:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w")

        threshold_var = tk.IntVar(value=128)
        threshold_scale = tk.Scale(
            parent,
            from_=0,
            to=255,
            variable=threshold_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
        )
        threshold_scale.pack(fill="x", pady=5)

        self.segmentation_params["threshold"] = threshold_var.get()
        threshold_scale.configure(
            command=lambda v: self.segmentation_params.update({"threshold": int(v)})
        )

    def _create_adaptive_params(self, parent):
        """Crea controles para parámetros de umbral adaptativo"""
        tk.Label(
            parent,
            text="Tamaño de bloque:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w")

        block_size_var = tk.IntVar(value=11)
        block_scale = tk.Scale(
            parent,
            from_=3,
            to=21,
            variable=block_size_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
        )
        block_scale.pack(fill="x", pady=5)

        self.segmentation_params["block_size"] = block_size_var.get()
        block_scale.configure(
            command=lambda v: self.segmentation_params.update({"block_size": int(v)})
        )

    def _create_kmeans_params(self, parent):
        """Crea controles para parámetros de K-means"""
        tk.Label(
            parent,
            text="Número de clusters (k):",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w")

        k_var = tk.IntVar(value=3)
        k_scale = tk.Scale(
            parent,
            from_=2,
            to=8,
            variable=k_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
        )
        k_scale.pack(fill="x", pady=5)

        self.segmentation_params["k"] = k_var.get()
        k_scale.configure(
            command=lambda v: self.segmentation_params.update({"k": int(v)})
        )

    def _create_morphological_params(self, parent):
        """Crea controles para parámetros morfológicos"""
        # Operación
        tk.Label(
            parent,
            text="Operación:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w")

        op_var = tk.StringVar(value="erode")
        op_frame = tk.Frame(parent, bg=self.theme["panel_bg"])
        op_frame.pack(fill="x", pady=5)

        for op in ["erode", "dilate", "open", "close"]:
            tk.Radiobutton(
                op_frame,
                text=op.capitalize(),
                variable=op_var,
                value=op,
                bg=self.theme["panel_bg"],
                fg=self.theme["text_color"],
                selectcolor=self.theme["accent"],
            ).pack(side="left", padx=5)

        # Tamaño del kernel
        tk.Label(
            parent,
            text="Tamaño del kernel:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", pady=(10, 0))

        kernel_var = tk.IntVar(value=3)
        kernel_scale = tk.Scale(
            parent,
            from_=3,
            to=15,
            variable=kernel_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
        )
        kernel_scale.pack(fill="x", pady=5)

        self.segmentation_params["operation"] = op_var.get()
        self.segmentation_params["kernel_size"] = kernel_var.get()

        op_var.trace(
            "w",
            lambda *args: self.segmentation_params.update({"operation": op_var.get()}),
        )
        kernel_scale.configure(
            command=lambda v: self.segmentation_params.update({"kernel_size": int(v)})
        )

    def _calculate_segmentation_properties(self, binary_image: np.ndarray) -> dict:
        """Calcula propiedades de la segmentación (área, perímetro)"""
        import cv2

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        properties = {
            "num_regions": len(contours),
            "total_area": 0,
            "total_perimeter": 0,
            "regions": [],
        }

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            properties["total_area"] += area
            properties["total_perimeter"] += perimeter
            properties["regions"].append(
                {"id": i + 1, "area": area, "perimeter": perimeter}
            )

        return properties
