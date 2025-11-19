import os

import yaml

VERSION_PATH = os.path.join("system-version.yaml")


class VersionController:
    def __init__(self):
        self.data = self._load_version_file()

    def _load_version_file(self):
        if not os.path.exists(VERSION_PATH):
            raise FileNotFoundError("No se encontró system-version.yaml")

        with open(VERSION_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_full_version(self):
        name = self.data.get("name", "APP")
        version = self.data.get("version", "0.0")
        release = self.data.get("release", "")
        build = self.data.get("build", None)

        build_text = f" (Build {build})" if build else ""
        release_text = f" {release}" if release else ""

        return f"{name} V{version}{release_text}{build_text}"

    def get_name(self):
        return self.data.get("name", "APP")

    def get_version(self):
        return self.data.get("version", "0.0")

    def get_release(self):
        return self.data.get("release", "")

    def get_author(self):
        return self.data.get("author", {})

    def get_raw(self):
        return self.data
