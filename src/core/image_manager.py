import numpy as np
import pydicom
from PIL import Image


class ImageManager:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.image_path = None
        self.is_dicom = False
        self.dicom_metadata = None

    def load_image(self, file_path):
        try:
            self.image_path = file_path

            if file_path.lower().endswith(".dcm"):
                self.is_dicom = True
                return self._load_dicom(file_path)
            else:
                self.is_dicom = False
                return self._load_standard_image(file_path)

        except Exception as e:
            return False, f"Error al cargar imagen: {str(e)}"

    def _load_dicom(self, file_path):
        try:
            dataset = pydicom.dcmread(file_path)
            self.dicom_metadata = dataset

            image_array = dataset.pixel_array
            # Normalizar a 8-bit
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
            self.original_image = Image.fromarray(image_array)
            self.current_image = self.original_image.copy()

            return True, "Imagen DICOM cargada correctamente"

        except ImportError:
            return False, "pydicom no está instalado para cargar archivos DICOM"
        except Exception as e:
            return False, f"Error al cargar DICOM: {str(e)}"

    def _load_standard_image(self, file_path):
        try:
            self.original_image = Image.open(file_path)

            if self.original_image.mode != "L":
                self.original_image = self.original_image.convert("L")

            self.current_image = self.original_image.copy()
            return True, "Imagen cargada correctamente"

        except Exception as e:
            return False, f"Error al cargar imagen: {str(e)}"

    def restore_original(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            return True
        return False

    def get_current_image(self):
        return self.current_image

    def update_image(self, new_image):
        if new_image and isinstance(new_image, Image.Image):
            self.current_image = new_image
            return True
        return False
