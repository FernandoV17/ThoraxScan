import tkinter as tk
from typing import Callable

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class FiltersPanel:
    def __init__(self, parent, theme: dict, fonts: dict, on_apply_filter: Callable):
        self.parent = parent
        self.theme = theme
        self.fonts = fonts
        self.on_apply_filter = on_apply_filter

        self.create_panel()
        logger.info("FiltersPanel inicializado")

    def create_panel(self):
        self.main_frame = tk.Frame(self.parent, bg=self.theme["panel_bg"])
        self.main_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            self.main_frame,
            text="Filtros de Imagen",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 12, "bold"),
        ).pack(anchor="nw", pady=(0, 10))

        self.create_filters_grid()

    def create_filters_grid(self):
        filters_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        filters_frame.pack(fill="x")

        basic_filters = [
            ("Enfoque", "sharpen"),
            ("Desenfoque", "blur"),
            ("Bordes", "edge_enhance"),
            ("Suavizar", "smooth"),
            ("Detalle", "detail"),
            ("Relieve", "emboss"),
        ]

        # Crear botones en grid 2x3
        row, col = 0, 0
        for name, filter_type in basic_filters:
            btn = tk.Button(
                filters_frame,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                bd=0,
                width=8,
                command=lambda ft=filter_type: self.on_apply_filter(ft),
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")

            col += 1
            if col > 2:
                col = 0
                row += 1

        filters_frame.columnconfigure(0, weight=1)
        filters_frame.columnconfigure(1, weight=1)
        filters_frame.columnconfigure(2, weight=1)

    def show_advanced_filters(self):
        adv_window = tk.Toplevel(self.parent)
        adv_window.title("Filtros Avanzados")
        adv_window.geometry("300x400")
        adv_window.configure(bg=self.theme["panel_bg"])

        advanced_filters = [
            ("Desenfoque Gaussiano", "gaussian_blur"),
            ("Máscara de Enfoque", "unsharp_mask"),
            ("Contorno", "contour"),
            ("Encontrar Bordes", "find_edges"),
            ("Mediana", "median"),
            ("Moda", "mode"),
        ]

        for name, filter_type in advanced_filters:
            tk.Button(
                adv_window,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                bd=0,
                command=lambda ft=filter_type: self.apply_advanced_filter(
                    ft, adv_window
                ),
            ).pack(fill="x", padx=10, pady=2)

    def apply_advanced_filter(self, filter_type: str, window: tk.Toplevel):
        self.on_apply_filter(filter_type)
        window.destroy()
