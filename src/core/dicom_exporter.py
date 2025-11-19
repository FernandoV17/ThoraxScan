import datetime
from typing import Optional

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class DICOMExporter:
    def __init__(self):
        logger.info("DICOMExporter inicializado")

    def export_as_dicom(
        self,
        image: Image.Image,
        original_dicom: Optional[Dataset] = None,
        output_path: str = "exported_image.dcm",
    ) -> bool:
        try:
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()

            ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

            ds.PatientName = "Export^ThoraxScan"
            ds.PatientID = "000000"
            ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
            ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
            ds.StudyDescription = "ThoraxScan Export"
            ds.Modality = "SC"

            if original_dicom:
                self._copy_dicom_metadata(original_dicom, ds)

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 7  # Para 8-bit
            ds.BitsStored = 8
            ds.BitsAllocated = 8

            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2).astype(np.uint8)

            ds.Rows, ds.Columns = img_array.shape
            ds.PixelData = img_array.tobytes()

            ds.save_as(output_path, write_like_original=False)
            logger.info(f"Imagen exportada como DICOM: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exportando DICOM: {e}")
            return False

    def _copy_dicom_metadata(self, source: Dataset, target: Dataset):
        try:
            fields_to_copy = [
                "PatientName",
                "PatientID",
                "PatientBirthDate",
                "PatientSex",
                "StudyInstanceUID",
                "StudyID",
                "StudyDate",
                "StudyTime",
                "SeriesInstanceUID",
                "SeriesNumber",
                "SeriesDate",
                "SeriesTime",
                "InstanceNumber",
                "BodyPartExamined",
                "ViewPosition",
            ]

            for field in fields_to_copy:
                if hasattr(source, field):
                    setattr(target, field, getattr(source, field))

        except Exception as e:
            logger.warning(f"Error copiando metadatos DICOM: {e}")
