import tkinter as tk
from tkinter import simpledialog
from typing import Callable

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class AdjustmentsPanel:
    def __init__(
        self,
        parent,
        theme: dict,
        fonts: dict,
        on_slider_change: Callable,
        on_reset: Callable,
        on_gamma: Callable,
        on_levels: Callable,
        on_normalize: Callable,
    ):
        self.parent = parent
        self.theme = theme
        self.fonts = fonts
        self.on_slider_change = on_slider_change
        self.on_reset = on_reset
        self.on_gamma = on_gamma
        self.on_levels = on_levels
        self.on_normalize = on_normalize

        self.brightness_var = None
        self.contrast_var = None
        self.brightness_value = None
        self.contrast_value = None

        self.create_panel()
        logger.info("AdjustmentsPanel inicializado")

    def create_panel(self):
        self.main_frame = tk.Frame(self.parent, bg=self.theme["panel_bg"])
        self.main_frame.pack(fill="x", pady=(20, 0))

        tk.Label(
            self.main_frame,
            text="Ajustes Manuales",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 12, "bold"),
        ).pack(anchor="nw", pady=(0, 10))

        self.create_slider_controls()

        self.create_additional_controls()

    def create_slider_controls(self):
        reset_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        reset_frame.pack(fill="x", pady=(0, 10))

        tk.Button(
            reset_frame,
            text="↺ Reset Ajustes",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=self.on_reset,
        ).pack(side="right")

        self.create_slider_with_input("Brillo:", "brightness", 0.1, 3.0, 1.0)

        self.create_slider_with_input("Contraste:", "contrast", 0.1, 3.0, 1.0)

    def create_slider_with_input(
        self, label: str, var_name: str, min_val: float, max_val: float, default: float
    ):
        frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        frame.pack(fill="x", pady=5)

        tk.Label(
            frame, text=label, bg=self.theme["panel_bg"], fg=self.theme["text_color"]
        ).pack(side="left")

        var = tk.DoubleVar(value=default)
        setattr(self, f"{var_name}_var", var)

        scale = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            resolution=0.01,
            variable=var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
            length=150,
            command=self.on_slider_change,
        )
        scale.pack(side="left", fill="x", expand=True, padx=(10, 5))

        value_frame = tk.Frame(frame, bg=self.theme["panel_bg"])
        value_frame.pack(side="right", padx=5)

        value_label = tk.Label(
            value_frame,
            text=f"{default:.2f}",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            width=5,
        )
        value_label.pack(side="left")
        setattr(self, f"{var_name}_value", value_label)

        edit_btn = tk.Button(
            value_frame,
            text="Num",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=("Arial", 8),
            width=2,
            command=lambda: self.on_edit_value(var_name, min_val, max_val),
        )
        edit_btn.pack(side="left", padx=(2, 0))

    def on_edit_value(self, var_name: str, min_val: float, max_val: float):
        current_val = getattr(self, f"{var_name}_var").get()

        new_val = simpledialog.askfloat(
            f"Editar {var_name.capitalize()}",
            f"Ingrese valor ({min_val} - {max_val}):",
            initialvalue=current_val,
            minvalue=min_val,
            maxvalue=max_val,
        )

        if new_val is not None:
            getattr(self, f"{var_name}_var").set(new_val)
            self.update_slider_values()
            self.on_slider_change()

    def create_additional_controls(self):
        # Frame para controles extra
        extra_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        extra_frame.pack(fill="x", pady=10)

        tk.Button(
            extra_frame,
            text="Ajustar Gamma",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=self.on_gamma,
        ).pack(side="left", padx=2)

        tk.Button(
            extra_frame,
            text="Niveles",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=self.on_levels,
        ).pack(side="left", padx=2)

        tk.Button(
            extra_frame,
            text="Normalizar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=self.on_normalize,
        ).pack(side="left", padx=2)

    def update_slider_values(self):
        if self.brightness_value and self.contrast_value:
            self.brightness_value.config(text=f"{self.brightness_var.get():.2f}")
            self.contrast_value.config(text=f"{self.contrast_var.get():.2f}")

    def reset_sliders(self):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.update_slider_values()

    def get_values(self) -> tuple:
        return self.brightness_var.get(), self.contrast_var.get()
