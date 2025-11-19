import json
import os


class UIStyleManager:
    def __init__(self):
        path = os.path.join("assets", "ui", "styles.json")
        with open(path, "r") as f:
            self.styles = json.load(f)

    def get(self, key_path):
        keys = key_path.split(".")
        value = self.styles
        for k in keys:
            value = value.get(k, None)
            if value is None:
                return None
        return value

    def get_font(self, category="global"):
        fam = self.get(f"{category}.font_family") or self.get("global.font_family")
        size = self.get(f"{category}.font_size") or self.get("global.font_size")
        weight = self.get(f"{category}.font_weight") or "normal"
        return (fam, size, weight)

    def apply_background(self, widget, category="panel"):
        color = self.get(f"{category}.bg_color") or self.get("global.bg_color")
        widget.configure(bg=color)

    def apply_button_style(self, button):
        btn = self.styles["button"]
        button.configure(
            bg=btn["bg_color"],
            fg=btn["fg_color"],
            font=(self.styles["global"]["font_family"], btn["font_size"]),
        )
