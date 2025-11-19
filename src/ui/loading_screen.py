import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class LoadingScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.overrideredirect(True)
        self.geometry("450x300+600+300")
        self.configure(bg="#1f1f1f")

        logo_path = os.path.join("assets", "icons", "logo.png")
        img = Image.open(logo_path).resize((160, 160))
        self.logo = ImageTk.PhotoImage(img)

        logo_label = tk.Label(self, image=self.logo, bg="#1f1f1f")
        logo_label.pack(pady=10)

        label = tk.Label(
            self,
            text="Inicializando el sistema...",
            font=("Segoe UI", 13),
            fg="white",
            bg="#1f1f1f",
        )
        label.pack(pady=5)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(padx=30, fill="x", pady=20)
        self.progress.start(10)
