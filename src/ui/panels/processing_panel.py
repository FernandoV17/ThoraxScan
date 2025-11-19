import tkinter as tk
from tkinter import messagebox
from typing import Callable

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class ProcessingPanel:
    def __init__(
        self,
        parent,
        theme: dict,
        fonts: dict,
        on_auto_enhance: Callable,
        on_restore: Callable,
    ):
        self.parent = parent
        self.theme = theme
        self.fonts = fonts
        self.on_auto_enhance = on_auto_enhance
        self.on_restore = on_restore

        self.create_panel()
        logger.info("ProcessingPanel inicializado")

    def create_panel(self):
        self.main_frame = tk.Frame(self.parent, bg=self.theme["panel_bg"])
        self.main_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            self.main_frame,
            text="Procesamiento Automático",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 12, "bold"),
        ).pack(anchor="nw", pady=(0, 10))

        self.create_processing_buttons()

    def create_processing_buttons(self):
        tk.Button(
            self.main_frame,
            text="Auto-mejorar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_auto_enhance,
        ).pack(fill="x", pady=3)

        tk.Button(
            self.main_frame,
            text="Restaurar Original",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_restore,
        ).pack(fill="x", pady=3)

        separator = tk.Frame(self.main_frame, height=2, bg=self.theme["accent"])
        separator.pack(fill="x", pady=10)

        tk.Label(
            self.main_frame,
            text="Métodos Específicos:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10),
        ).pack(anchor="w", pady=(0, 5))

        methods_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        methods_frame.pack(fill="x")

        methods = [
            ("Mejorar Contraste", "contrast"),
            ("Reducir Ruido", "denoise"),
            ("Enfocar", "sharpness"),
        ]

        for name, method in methods:
            tk.Button(
                methods_frame,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                bd=0,
                command=lambda m=method: self.on_specific_method(m),
            ).pack(fill="x", pady=2)

    def on_specific_method(self, method: str):
        messagebox.showinfo("Procesamiento", f"Método {method} en desarrollo")
