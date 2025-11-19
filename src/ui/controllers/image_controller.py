import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Callable, Optional

from src.core.dicom_exporter import DICOMExporter
from src.core.image_manager import ImageManager
from src.core.metadata_extractor import MetadataExtractor
from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class ImageController:
    def __init__(self):
        self.image_manager = ImageManager()
        self.metadata_extractor = MetadataExtractor()
        self.dicom_exporter = DICOMExporter()
        self.manual_adjustments = None
        self.analysis_controller = None

        self._callbacks = {}
        logger.info("ImageController inicializado")

    def set_manual_adjustments(self, manual_adjustments):
        self.manual_adjustments = manual_adjustments

    def set_analysis_controller(self, analysis_controller):
        self.analysis_controller = analysis_controller

    def register_callback(self, event: str, callback: Callable):
        self._callbacks[event] = callback

    def _emit(self, event: str, *args):
        if event in self._callbacks:
            self._callbacks[event](*args)

    def open_image(self):
        file_types = [
            ("Imágenes médicas", "*.dcm"),
            ("Imágenes", "*.png *.jpg *.jpeg *.tiff *.bmp"),
            ("Todos los archivos", "*.*"),
        ]

        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de rayos X", filetypes=file_types
        )

        if file_path:
            try:
                success, message = self.image_manager.load_image(file_path)
                if success:
                    self._emit("image_loaded")
                    return True, message
                else:
                    return False, message
            except Exception as e:
                return False, f"Error al cargar imagen: {str(e)}"
        return False, "No se seleccionó archivo"

    def auto_enhance(self):
        if not self.image_manager.has_image():
            return False, "Primero carga una imagen"

        if not self.manual_adjustments:
            return False, "Sistema de ajustes no disponible"

        try:
            enhanced_image = self.manual_adjustments.enhance(
                self.image_manager.current_image
            )
            self.image_manager.update_image(enhanced_image, "Auto-mejorada")
            self._emit("image_updated")
            return True, "Imagen mejorada automáticamente"
        except Exception as e:
            return False, f"No se pudo mejorar la imagen: {str(e)}"

    def restore_original(self):
        if self.image_manager.original_image:
            self.image_manager.restore_original()
            self._emit("image_updated")
            return True, "Imagen restaurada a su estado original"
        else:
            return False, "No hay imagen original para restaurar"

    def apply_brightness_contrast(self, brightness: float, contrast: float):
        if not self.image_manager.has_image():
            return False

        if not self.manual_adjustments:
            return False

        try:
            original_image = self.image_manager.original_image
            adjusted = self.manual_adjustments.adjust_brightness_contrast(
                original_image, brightness, contrast
            )
            self.image_manager.update_image(
                adjusted, f"Ajuste - B:{brightness:.2f} C:{contrast:.2f}"
            )
            self._emit("image_updated")
            return True
        except Exception as e:
            logger.error(f"Error aplicando brillo/contraste: {e}")
            return False

    def apply_filter(self, filter_type: str):
        if not self.image_manager.has_image():
            return False

        if not self.manual_adjustments:
            return False

        try:
            current_image = self.image_manager.get_current_image()
            filtered = self.manual_adjustments.apply_filter(current_image, filter_type)
            self.image_manager.update_image(filtered, f"Filtro: {filter_type}")
            self._emit("image_updated")
            return True
        except Exception as e:
            logger.error(f"Error aplicando filtro {filter_type}: {e}")
            return False

    def adjust_gamma(self):
        if not self.image_manager.has_image():
            return False, "Primero carga una imagen"

        if not self.manual_adjustments:
            return False, "Sistema de ajustes no disponible"

        gamma = simpledialog.askfloat(
            "Ajuste de Gamma",
            "Ingrese valor gamma (0.1 - 5.0):",
            initialvalue=1.0,
            minvalue=0.1,
            maxvalue=5.0,
        )

        if gamma is not None:
            try:
                current_image = self.image_manager.get_current_image()
                adjusted = self.manual_adjustments.adjust_gamma(current_image, gamma)
                self.image_manager.update_image(adjusted, f"Gamma: {gamma:.2f}")
                self._emit("image_updated")
                return True, f"Gamma ajustado a {gamma:.2f}"
            except Exception as e:
                return False, f"No se pudo ajustar gamma: {str(e)}"
        return False, "Operación cancelada"

    def normalize_image(self):
        if not self.image_manager.has_image():
            return False, "Primero carga una imagen"

        if not self.manual_adjustments:
            return False, "Sistema de ajustes no disponible"

        try:
            current_image = self.image_manager.get_current_image()
            normalized = self.manual_adjustments.normalize_image(
                current_image, "global"
            )
            self.image_manager.update_image(normalized, "Normalizada")
            self._emit("image_updated")
            return True, "Imagen normalizada"
        except Exception as e:
            return False, f"No se pudo normalizar: {str(e)}"

    def export_image(self):
        if not self.image_manager.has_image():
            return False, "Primero carga una imagen"

        file_types = [
            ("DICOM", "*.dcm"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("TIFF", "*.tiff"),
            ("BMP", "*.bmp"),
        ]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=file_types,
            title="Exportar imagen como...",
        )

        if file_path:
            try:
                if file_path.endswith(".dcm"):
                    original_dicom = getattr(self.image_manager, "dicom_metadata", None)
                    success = self.dicom_exporter.export_as_dicom(
                        self.image_manager.current_image, original_dicom, file_path
                    )
                    if success:
                        return True, f"Imagen exportada como DICOM: {file_path}"
                    else:
                        return False, "No se pudo exportar como DICOM"
                else:
                    self.image_manager.current_image.save(file_path)
                    return True, f"Imagen exportada a: {file_path}"
            except Exception as e:
                return False, f"No se pudo exportar: {str(e)}"
        return False, "Exportación cancelada"

    def get_metadata(self):
        if self.image_manager.has_image():
            return self.metadata_extractor.extract_metadata(self.image_manager)
        return {}

    def get_histogram_data(self):
        if self.image_manager.has_image() and self.manual_adjustments:
            current_image = self.image_manager.get_current_image()
            return self.manual_adjustments.calculate_histogram(current_image)
        return {}

    def get_current_image(self):
        return self.image_manager.get_current_image()

    def has_image(self):
        return self.image_manager.has_image()
