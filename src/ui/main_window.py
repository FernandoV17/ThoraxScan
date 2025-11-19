import json
import os
import tkinter as tk

from PIL import Image, ImageTk

from src.ui.controllers.version_controller import VersionController
from src.ui.widgets.welcome_popup import WelcomePopup

STYLES_PATH = os.path.join("assets", "ui", "styles.json")
LOGO_PATH = os.path.join("assets", "icons", "logo.png")


class MainWindow:
    def __init__(self, root):
        self.root = root  # ← DEBE IR PRIMERO
        self.version = VersionController()

        self.root.title(self.version.get_full_version())

        self.root.geometry("1300x850")

        self.load_styles()
        self.apply_global_style()

        self.create_top_bar()
        self.create_main_layout()

        self.root.after(
            300, lambda: WelcomePopup(self.root, self.theme, self.fonts, self.version)
        )

    def load_styles(self):
        if os.path.exists(STYLES_PATH):
            with open(STYLES_PATH, "r", encoding="utf-8") as f:
                self.styles = json.load(f)
        else:
            raise FileNotFoundError("No se encontró styles.json en assets/ui/")

        theme_name = self.styles.get("current_theme", "dark")
        self.theme = self.styles["themes"][theme_name]
        self.fonts = self.styles["fonts"]

    def apply_global_style(self):
        self.root.configure(bg=self.theme["global_bg"])

    def create_top_bar(self):
        top_bar = tk.Frame(self.root, bg=self.theme["panel_bg"], height=70)
        top_bar.pack(side="top", fill="x")

        try:
            logo_img = Image.open(LOGO_PATH).resize((55, 55))
            self.logo_img = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(
                top_bar, image=self.logo_img, bg=self.theme["panel_bg"]
            )
        except Exception:
            logo_label = tk.Label(
                top_bar,
                text="[LOGO]",
                fg=self.theme["text_color"],
                bg=self.theme["panel_bg"],
            )
        logo_label.pack(side="left", padx=12)

        # Ribbon (simulación tipo Windows)
        ribbon = tk.Frame(top_bar, bg=self.theme["panel_bg"])
        ribbon.pack(side="left", padx=10)

        btn_cfg = {
            "bg": self.theme["accent"],
            "fg": self.theme["text_color"],
            "bd": 0,
            "padx": 10,
            "pady": 6,
            "font": (self.fonts["main"]["family"], self.fonts["main"]["size"]),
        }

        tk.Button(ribbon, text="Archivo", command=self.on_file, **btn_cfg).pack(
            side="left", padx=4
        )
        tk.Button(
            ribbon, text="Procesamiento", command=self.on_processing, **btn_cfg
        ).pack(side="left", padx=4)
        tk.Button(ribbon, text="Análisis", command=self.on_analysis, **btn_cfg).pack(
            side="left", padx=4
        )
        tk.Button(ribbon, text="Ajustes", command=self.on_settings, **btn_cfg).pack(
            side="left", padx=4
        )

    def create_main_layout(self):
        main_frame = tk.Frame(self.root, bg=self.theme["global_bg"])
        main_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(
            main_frame, width=300, bg=self.theme["panel_bg"], padx=10, pady=10
        )
        self.left_panel.pack(side="left", fill="y")

        tk.Label(
            self.left_panel,
            text="Metadatos",
            fg=self.theme["text_color"],
            bg=self.theme["panel_bg"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        ).pack(anchor="nw")

        self.meta_box = tk.Text(
            self.left_panel,
            height=14,
            width=32,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            bd=0,
        )
        self.meta_box.insert("1.0", "(Metadatos se cargarán aquí...)")
        self.meta_box.config(state="disabled")
        self.meta_box.pack(pady=8)

        tk.Label(
            self.left_panel,
            text="Opciones básicas",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 12, "bold"),
        ).pack(anchor="nw", pady=(10, 6))

        tk.Button(
            self.left_panel,
            text="Abrir imagen...",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_open_image,
        ).pack(fill="x", pady=3)

        tk.Button(
            self.left_panel,
            text="Auto-mejorar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_auto_enhance,
        ).pack(fill="x", pady=3)

        tk.Button(
            self.left_panel,
            text="Restaurar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_restore,
        ).pack(fill="x", pady=3)

        # Panel derecho: Avanzado
        self.right_panel = tk.Frame(
            main_frame, width=300, bg=self.theme["panel_bg"], padx=10, pady=10
        )
        self.right_panel.pack(side="right", fill="y")

        tk.Label(
            self.right_panel,
            text="Opciones avanzadas",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        ).pack(anchor="ne")

        tk.Button(
            self.right_panel,
            text="Detección de anomalías",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_detect_anomalies,
        ).pack(fill="x", pady=3)

        tk.Button(
            self.right_panel,
            text="Fracturas (demo)",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_detect_fractures,
        ).pack(fill="x", pady=3)

        tk.Button(
            self.right_panel,
            text="Cardiomegalia (demo)",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.on_detect_heart,
        ).pack(fill="x", pady=3)

        self.center_panel = tk.Frame(main_frame, bg=self.theme["global_bg"])
        self.center_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.center_panel, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(200, 200, text="AQUÍ VA LA IMAGEN", fill="white")

        self.hist_frame = tk.Frame(
            self.center_panel, bg=self.theme["panel_bg"], height=150, width=380
        )
        self.hist_frame.pack(side="bottom", anchor="sw", pady=10, padx=10)

        tk.Label(
            self.hist_frame,
            text="Histograma",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack()

    # ================LOGICA============================

    def on_file(self):
        pass

    def on_processing(self):
        pass

    def on_analysis(self):
        pass

    def on_settings(self):
        pass

    def on_open_image(self):
        pass

    def on_auto_enhance(self):
        pass

    def on_restore(self):
        pass

    def on_detect_anomalies(self):
        pass

    def on_detect_fractures(self):
        pass

    def on_detect_heart(self):
        pass

    def show_welcome_message(self):
        WelcomePopup(self.root, self.theme, self.fonts)
