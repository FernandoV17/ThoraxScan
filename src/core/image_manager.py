import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from src.helpers.logger import get_module_logger, log_function_call

logger = get_module_logger(__name__)


class ImageManager:
    def __init__(self):
        self.original_image: Optional[Image.Image] = None
        self.current_image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        self.is_dicom: bool = False
        self.dicom_metadata: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self.load_time: float = 0
        self.processing_history = []

        logger.info("ImageManager inicializado")

    @log_function_call(log_args=True, log_result=True)
    def load_image(self, file_path: str) -> Tuple[bool, str]:
        start_time = time.time()

        with self._lock:
            try:
                self.image_path = file_path
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == ".dcm":
                    success, message = self._load_dicom(file_path)
                else:
                    success, message = self._load_standard_image(file_path)

                if success:
                    self.load_time = time.time() - start_time
                    logger.info(f"Imagen cargada: {file_path} ({self.load_time:.2f}s)")
                    self.processing_history.clear()
                    self.processing_history.append("Imagen original cargada")

                return success, message

            except Exception as e:
                error_msg = f"Error crítico al cargar imagen: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

    def _load_dicom(self, file_path: str) -> Tuple[bool, str]:
        try:
            import pydicom
            from pydicom.pixel_data_handlers import apply_voi_lut

            dataset = pydicom.dcmread(file_path)
            self.dicom_metadata = self._extract_dicom_metadata(dataset)

            if "VOILUTSequence" in dataset:
                image_array = apply_voi_lut(dataset.pixel_array, dataset)
            else:
                image_array = dataset.pixel_array

            if image_array.dtype != np.uint8:
                image_array = self._normalize_dicom_array(image_array)

            self.original_image = Image.fromarray(image_array)
            self.current_image = self.original_image.copy()
            self.is_dicom = True

            logger.debug(f"DICOM cargado: {image_array.shape}, {image_array.dtype}")
            return True, "Imagen DICOM cargada correctamente"

        except ImportError:
            return False, "pydicom no está instalado. Ejecuta: pip install pydicom"
        except Exception as e:
            logger.error(f"Error cargando DICOM: {e}")
            return False, f"Error DICOM: {str(e)}"

    def _load_standard_image(self, file_path: str) -> Tuple[bool, str]:
        try:
            self.original_image = Image.open(file_path)

            if self.original_image.mode != "L":
                self.original_image = self.original_image.convert("L")

            self.current_image = self.original_image.copy()
            self.is_dicom = False
            self.dicom_metadata = None

            logger.debug(
                f"Imagen estándar cargada: {self.original_image.size}, {self.original_image.mode}"
            )
            return True, "Imagen cargada correctamente"

        except Exception as e:
            logger.error(f"Error cargando imagen estándar: {e}")
            return False, f"Error imagen: {str(e)}"

    def _normalize_dicom_array(self, image_array: np.ndarray) -> np.ndarray:
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = (image_array * 255).astype(np.uint8)  # 8bits
        else:
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            if max_val > min_val:
                image_array = (
                    (image_array - min_val) / (max_val - min_val) * 255
                ).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        return image_array

    def _extract_dicom_metadata(self, dataset) -> Dict[str, Any]:
        metadata = {}
        try:
            dicom_fields = {
                "PatientName": "Nombre Paciente",
                "PatientID": "ID Paciente",
                "StudyDate": "Fecha Estudio",
                "Modality": "Modalidad",
                "StudyDescription": "Descripción",
                "BodyPartExamined": "Área Examinada",
                "SeriesDescription": "Serie",
                "Manufacturer": "Fabricante",
            }

            for field, description in dicom_fields.items():
                if hasattr(dataset, field):
                    value = getattr(dataset, field)
                    metadata[description] = str(value)

        except Exception as e:
            logger.warning(f"Error extrayendo metadatos DICOM: {e}")

        return metadata

    @log_function_call()
    def restore_original(self) -> bool:
        """Restaura la imagen original"""
        with self._lock:
            if self.original_image:
                self.current_image = self.original_image.copy()
                self.processing_history.clear()
                self.processing_history.append("Imagen restaurada a original")
                logger.info("Imagen restaurada a estado original")
                return True
            logger.warning("No hay imagen original para restaurar")
            return False

    @log_function_call(log_result=False)
    def update_image(
        self, new_image: Image.Image, operation: str = "Operación"
    ) -> bool:
        with self._lock:
            if new_image and isinstance(new_image, Image.Image):
                self.current_image = new_image
                self.processing_history.append(operation)
                logger.debug(f"Imagen actualizada: {operation}")
                return True
            return False

    def get_image_info(self) -> Dict[str, Any]:
        with self._lock:
            if not self.current_image:
                return {}

            return {
                "size": self.current_image.size,
                "mode": self.current_image.mode,
                "format": "DICOM" if self.is_dicom else "Estándar",
                "history": self.processing_history.copy(),
                "load_time": f"{self.load_time:.2f}s",
            }

    def get_current_image(self) -> Optional[Image.Image]:
        with self._lock:
            return self.current_image.copy() if self.current_image else None

    def has_image(self) -> bool:
        return self.current_image is not None
