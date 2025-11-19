import datetime
import os


class MetadataExtractor:
    def __init__(self):
        pass

    def extract_metadata(self, image_manager):
        if not image_manager.current_image:
            return {"Estado": "No hay imagen cargada"}

        metadata = {}

        if image_manager.current_image:
            metadata["Tamaño"] = f"{image_manager.current_image.size}"
            metadata["Modo"] = image_manager.current_image.mode
            metadata["Formato"] = getattr(
                image_manager.current_image, "format", "Desconocido"
            )

        if image_manager.image_path:
            file_info = self._get_file_info(image_manager.image_path)
            metadata.update(file_info)

        if image_manager.is_dicom and image_manager.dicom_metadata:
            dicom_info = self._extract_dicom_metadata(image_manager.dicom_metadata)
            metadata.update(dicom_info)
        else:
            metadata["Tipo"] = "Imagen estándar"

        return metadata

    def _get_file_info(self, file_path):
        try:
            stat = os.stat(file_path)
            file_size = stat.st_size / 1024

            return {
                "Archivo": os.path.basename(file_path),
                "Ruta": file_path,
                "Tamaño archivo": f"{file_size:.2f} KB",
                "Modificado": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        except:
            return {"Archivo": "Información no disponible"}

    def _extract_dicom_metadata(self, dataset):
        """Extraer metadatos DICOM"""
        dicom_info = {"Tipo": "DICOM"}

        try:
            if hasattr(dataset, "PatientName"):
                dicom_info["Paciente"] = str(dataset.PatientName)
            if hasattr(dataset, "PatientID"):
                dicom_info["ID Paciente"] = str(dataset.PatientID)
            if hasattr(dataset, "StudyDate"):
                dicom_info["Fecha estudio"] = str(dataset.StudyDate)
            if hasattr(dataset, "Modality"):
                dicom_info["Modalidad"] = str(dataset.Modality)
            if hasattr(dataset, "StudyDescription"):
                dicom_info["Descripción"] = str(dataset.StudyDescription)

        except:
            dicom_info["Metadatos DICOM"] = "No disponibles"

        return dicom_info
