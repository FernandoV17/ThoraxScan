import json
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
import threading
from tkinter import ttk
from PIL import Image, ImageTk

from src.helpers.logger import get_module_logger
from src.processing.auto_enhancer import AutoEnhancer
from src.processing.manual_adjustments import ManualAdjustments
from src.ui.controllers.analysis_controller import AnalysisController
from src.ui.controllers.image_controller import ImageController
from src.ui.controllers.version_controller import VersionController
from src.ui.panels.adjustments_panel import AdjustmentsPanel
from src.ui.panels.filters_panel import FiltersPanel
from src.ui.panels.histogram_panel import HistogramPanel
from src.ui.panels.processing_panel import ProcessingPanel
from src.ui.widgets.welcome_popup import WelcomePopup

logger = get_module_logger(__name__)

STYLES_PATH = os.path.join("assets", "ui", "styles.json")
LOGO_PATH = os.path.join("assets", "icons", "logo.png")


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.version = VersionController()

        self.manual_adjustments = ManualAdjustments()
        self.auto_enhancer = AutoEnhancer()

        self.image_controller = ImageController()
        self.analysis_controller = AnalysisController()

        self.histogram_panel = None

        self.image_controller.set_manual_adjustments(self.manual_adjustments)
        self.image_controller.set_analysis_controller(self.analysis_controller)

        self.guide_lines = []
        self.mouse_x = 0
        self.mouse_y = 0
        self.preview_image = None

        self.setup_ui()
        logger.info("MainWindow inicializado correctamente")

    def setup_ui(self):
        self.root.title(self.version.get_full_version())
        self.root.state("zoomed")

        self.load_styles()
        self.apply_global_style()
        self.create_top_bar()
        self.create_main_layout()
        self.create_panels()
        self.setup_callbacks()

        self.root.after(
            300, lambda: WelcomePopup(self.root, self.theme, self.fonts, self.version)
        )

    def setup_callbacks(self):
        self.image_controller.register_callback("image_loaded", self.on_image_loaded)
        self.image_controller.register_callback("image_updated", self.on_image_updated)

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
        except Exception as e:
            logger.warning(f"No se pudo cargar el logo: {e}")
            logo_label = tk.Label(
                top_bar,
                text="[LOGO]",
                fg=self.theme["text_color"],
                bg=self.theme["panel_bg"],
                font=(self.fonts["title"]["family"], 12, "bold"),
            )
        logo_label.pack(side="left", padx=12)

        title_label = tk.Label(
            top_bar,
            text="ThoraxScan - Analizador de Rayos X",
            fg=self.theme["text_color"],
            bg=self.theme["panel_bg"],
            font=(self.fonts["title"]["family"], 16, "bold"),
        )
        title_label.pack(side="left", padx=10)

        ribbon = tk.Frame(top_bar, bg=self.theme["panel_bg"])
        ribbon.pack(side="left", padx=20)

        btn_cfg = {
            "bg": self.theme["accent"],
            "fg": self.theme["text_color"],
            "bd": 0,
            "padx": 12,
            "pady": 8,
            "font": (self.fonts["main"]["family"], self.fonts["main"]["size"]),
            "relief": "flat",
            "cursor": "hand2",
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

        spacer = tk.Frame(top_bar, bg=self.theme["panel_bg"])
        spacer.pack(side="right", fill="x", expand=True)

        version_label = tk.Label(
            top_bar,
            text=f"v{self.version.get_version()}",
            fg=self.theme["text_color"],
            bg=self.theme["panel_bg"],
            font=(self.fonts["main"]["family"], 10),
            cursor="hand2",
        )
        version_label.pack(side="right", padx=12)
        version_label.bind("<Button-1>", lambda e: self.show_version_info())

        separator = tk.Frame(top_bar, height=2, bg=self.theme["accent"])
        separator.pack(side="bottom", fill="x")

    def create_main_layout(self):
        """Crea el layout principal con paneles izquierdo, central y derecho"""
        main_frame = tk.Frame(self.root, bg=self.theme["global_bg"])
        main_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(
            main_frame, width=320, bg=self.theme["panel_bg"], padx=12, pady=12
        )
        self.left_panel.pack(side="left", fill="y")
        self.left_panel.pack_propagate(False)

        meta_frame = tk.Frame(self.left_panel, bg=self.theme["panel_bg"])
        meta_frame.pack(fill="x", pady=(0, 15))

        meta_title = tk.Label(
            meta_frame,
            text="Metadatos",
            fg=self.theme["text_color"],
            bg=self.theme["panel_bg"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        )
        meta_title.pack(anchor="nw", pady=(0, 8))

        meta_border = tk.Frame(meta_frame, bg=self.theme["accent"], padx=1, pady=1)
        meta_border.pack(fill="x")

        self.meta_box = tk.Text(
            meta_border,
            height=12,
            width=35,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            bd=0,
            wrap="word",
            font=(self.fonts["main"]["family"], 9),
            relief="flat",
        )
        self.meta_box.pack(fill="both", expand=True)
        self.meta_box.insert(
            "1.0", "No hay imagen cargada\n\nSelecciona 'Abrir imagen' para comenzar"
        )
        self.meta_box.config(state="disabled")

        open_btn = tk.Button(
            self.left_panel,
            text="📁 Abrir imagen...",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=15,
            pady=8,
            font=(self.fonts["main"]["family"], 10, "bold"),
            cursor="hand2",
            command=self.on_open_image,
        )
        open_btn.pack(fill="x", pady=(0, 20))

        self.right_panel = tk.Frame(
            main_frame, width=280, bg=self.theme["panel_bg"], padx=12, pady=12
        )
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        right_title = tk.Label(
            self.right_panel,
            text="🔧 Análisis Avanzado",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        )
        right_title.pack(anchor="ne", pady=(0, 15))

        # --- Segmentación (Manual) Section ---
        seg_frame = tk.LabelFrame(
            self.right_panel,
            text="Segmentación (Manual)",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
)
        seg_frame.pack(fill="x", pady=4)

        # Umbral: Adaptativo (opens slider window) and Otsu (automatic)
        umbral_frame = tk.Frame(seg_frame, bg=self.theme["panel_bg"])
        umbral_frame.pack(fill="x", pady=(4, 2), padx=6)

        tk.Label(
            umbral_frame,
            text="Umbral:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        ).pack(anchor="w")

        tk.Button(
            umbral_frame,
            text="Adaptativo...",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=self.open_adaptive_segmentation_window,
        ).pack(fill="x", pady=2)

        tk.Button(
            umbral_frame,
            text="Binary...",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=self.open_binary_threshold_window,
        ).pack(fill="x", pady=2)

        tk.Button(
            umbral_frame,
            text="Área y Perímetro",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=self.on_current_threshold_stats,
        ).pack(fill="x", pady=2)

        tk.Button(
            umbral_frame,
            text="Otsu",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=lambda: self.apply_segmentation("otsu"),
        ).pack(fill="x", pady=2)

        # Cluster (semi-automática - ask for k)
        tk.Button(
            seg_frame,
            text="Cluster (k-means)",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=self.on_cluster_segmentation,
        ).pack(fill="x", pady=(6, 2), padx=6)

        tk.Button(
            seg_frame,
            text="Área y Perímetro (Clusters)",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=self.on_cluster_stats,
        ).pack(fill="x", pady=(6, 2), padx=6)

        # Morfológica
        morph_frame = tk.Frame(seg_frame, bg=self.theme["panel_bg"])
        morph_frame.pack(fill="x", pady=(6, 4), padx=6)

        tk.Label(
            morph_frame,
            text="Morfológica:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        ).pack(anchor="w")

        tk.Button(
            morph_frame,
            text="Erosionar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=lambda: self.apply_segmentation("erode"),
        ).pack(fill="x", pady=2)

        tk.Button(
            morph_frame,
            text="Dilatar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=8,
            pady=6,
            font=(self.fonts["main"]["family"], 9),
            cursor="hand2",
            command=lambda: self.apply_segmentation("dilate"),
        ).pack(fill="x", pady=2)

        # --- Detección fracturas (Automático) ---
        tk.Button(
            self.right_panel,
            text="Detección de Fracturas (Automático)",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            padx=10,
            pady=10,
            font=(self.fonts["main"]["family"], 10),
            cursor="hand2",
            command=self.on_detect_fractures,
        ).pack(fill="x", pady=(8, 4))

        self.center_panel = tk.Frame(
            main_frame, bg=self.theme["global_bg"], padx=10, pady=10
        )
        self.center_panel.pack(side="left", fill="both", expand=True)

        canvas_frame = tk.Frame(self.center_panel, bg="black")
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, bg="black", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)

        self.mouse_x = 0
        self.mouse_y = 0
        self.coord_text = None

        self.canvas.create_text(
            400,
            300,
            text="Área de Visualización\n\nAbre una imagen para comenzar el análisis",
            fill="white",
            font=("Arial", 14),
            justify="center",
        )

        status_frame = tk.Frame(self.center_panel, bg=self.theme["panel_bg"], height=25)
        status_frame.pack(fill="x", side="bottom", pady=(5, 0))
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Listo",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 9),
            anchor="w",
        )
        self.status_label.pack(side="left", padx=10)

        self.image_info_label = tk.Label(
            status_frame,
            text="Sin imagen cargada",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 9),
            anchor="e",
        )
        self.image_info_label.pack(side="right", padx=10)

        self.hist_frame = tk.Frame(
            self.center_panel, bg=self.theme["panel_bg"], height=200
        )
        self.hist_frame.pack(side="bottom", fill="x", pady=(5, 0))
        self.hist_frame.pack_propagate(False)

        self.histogram_panel = HistogramPanel(self.hist_frame, self.theme, self.fonts)

        hist_title = tk.Label(
            self.hist_frame,
            text="Histograma",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        )
        hist_title.pack(anchor="nw", padx=10, pady=5)

        hist_placeholder = tk.Label(
            self.hist_frame,
            text="El histograma se mostrará aquí cuando cargues una imagen",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 9),
        )
        hist_placeholder.pack(expand=True)

    def create_panels(self):
        self.processing_panel = ProcessingPanel(
            self.left_panel,
            self.theme,
            self.fonts,
            on_auto_enhance=self.on_auto_enhance,
            on_restore=self.on_restore,
        )

        self.filters_panel = FiltersPanel(
            self.left_panel,
            self.theme,
            self.fonts,
            on_apply_filter=self.on_apply_filter,
        )

        self.adjustments_panel = AdjustmentsPanel(
            self.left_panel,
            self.theme,
            self.fonts,
            on_slider_change=self.on_slider_change,
            on_reset=self.on_reset_sliders,
            on_gamma=self.on_gamma_adjust,
            on_levels=self.on_levels_adjust,
            on_normalize=self.on_normalize_image,
        )

    def on_image_loaded(self):
        self.display_image()
        self.update_metadata()
        self.update_histogram()
        self.update_status("Imagen cargada correctamente")
        self.update_image_info()

    def on_image_updated(self):
        self.display_image()
        self.update_histogram()
        self.update_image_info()

    def on_slider_change(self, value=None):
        self.adjustments_panel.update_slider_values()
        brightness, contrast = self.adjustments_panel.get_values()
        self.image_controller.apply_brightness_contrast(brightness, contrast)

    def on_reset_sliders(self):
        self.adjustments_panel.reset_sliders()
        success, message = self.image_controller.restore_original()
        if not success:
            self.show_message("Advertencia", message)

    def on_open_image(self):
        success, message = self.image_controller.open_image()
        if not success:
            self.show_message("Error", message)

    def on_auto_enhance(self):
        success, message = self.image_controller.auto_enhance()
        self.show_message("Éxito" if success else "Error", message)

    def on_restore(self):
        success, message = self.image_controller.restore_original()
        self.show_message("Éxito" if success else "Advertencia", message)

    def on_apply_filter(self, filter_type: str):
        self.image_controller.apply_filter(filter_type)

    def on_gamma_adjust(self):
        success, message = self.image_controller.adjust_gamma()
        if message != "Operación cancelada":
            self.show_message("Éxito" if success else "Error", message)

    def on_normalize_image(self):
        success, message = self.image_controller.normalize_image()
        self.show_message("Éxito" if success else "Error", message)

    def on_export_image(self):
        success, message = self.image_controller.export_image()
        if message != "Exportación cancelada":
            self.show_message("Éxito" if success else "Error", message)

    def display_image(self):
        # If a preview image is set, display it instead of current image
        if hasattr(self, "preview_image") and self.preview_image is not None:
            current_image = self.preview_image
        else:
            current_image = self.image_controller.get_current_image()
        if current_image:
            try:
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    img = current_image.copy()
                    img.thumbnail(
                        (canvas_width - 20, canvas_height - 20),
                        Image.Resampling.LANCZOS,
                    )

                    self.tk_image = ImageTk.PhotoImage(img)
                    self.canvas.delete("all")
                    self.canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=self.tk_image,
                        anchor="center",
                    )
            except Exception as e:
                logger.error(f"Error mostrando imagen: {e}")

    def update_metadata(self):
        metadata = self.image_controller.get_metadata()

        self.meta_box.config(state="normal")
        self.meta_box.delete("1.0", "end")

        for key, value in metadata.items():
            self.meta_box.insert("end", f"{key}: {value}\n")

        self.meta_box.config(state="disabled")

    def update_histogram(self):
        hist_data = self.image_controller.get_histogram_data()
        # Por implementar: integrar con HistogramPanel
        logger.debug("Histograma actualizado")

    def update_status(self, message: str):
        """Actualiza la barra de estado"""
        self.status_label.config(text=message)

    def update_image_info(self):
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            info = f"Tamaño: {image.size} | Modo: {image.mode}"
            self.image_info_label.config(text=info)
        else:
            self.image_info_label.config(text="Sin imagen cargada")

    def Segmentacion(self):
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            result = self.analysis_controller.detect_anomalies(image)
            self.show_analysis_result("Anomalías Detectadas", result)
        else:
            self.show_message("Advertencia", "Primero carga una imagen")

    def on_detect_fractures(self):
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            result = self.analysis_controller.detect_fractures(image)
            self.show_analysis_result("Fracturas Detectadas", result)
        else:
            self.show_message("Advertencia", "Primero carga una imagen")

    def on_detect_heart(self):
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            result = self.analysis_controller.detect_cardiomegaly(image)
            self.show_analysis_result("Cardiomegalia", result)
        else:
            self.show_message("Advertencia", "Primero carga una imagen")

    def show_analysis_result(self, title, result):
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.geometry("500x400")

        text = tk.Text(result_window, wrap="word", padx=10, pady=10)
        text.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(text)
        scrollbar.pack(side="right", fill="y")

        text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)

        text.insert("1.0", str(result))
        text.config(state="disabled")

    def on_file(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Abrir imagen...", command=self.on_open_image)
        menu.add_separator()
        menu.add_command(label="Guardar imagen", command=self.on_save_image)
        menu.add_command(label="Exportar como...", command=self.on_export_image)
        menu.add_separator()
        menu.add_command(label="Salir", command=self.root.quit)
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def on_processing(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Auto-mejorar", command=self.on_auto_enhance)
        menu.add_separator()
        menu.add_command(
            label="Ajustar brillo/contraste", command=self.on_adjust_brightness
        )
        menu.add_command(label="Filtros", command=self.on_filters_menu)
        menu.add_separator()
        menu.add_command(label="Restaurar original", command=self.on_restore)
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def on_analysis(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Segmentacion", command=self.Segmentacion)
        menu.add_command(
            label="Detección de fracturas", command=self.on_detect_fractures
        )
        menu.add_separator()
        menu.add_command(label="Mostrar histograma", command=self.on_show_histogram)
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def on_settings(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Tema claro/oscuro", command=self.on_toggle_theme)
        menu.add_command(label="Configuración", command=self.on_config)
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def show_message(self, title: str, message: str):
        if title.lower() == "error":
            messagebox.showerror(title, message)
        elif title.lower() == "advertencia":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)

    def show_version_info(self):
        messagebox.showinfo(
            "Información de Versión",
            f"ThoraxScan\n\n"
            f"Versión: {self.version.get_full_version()}\n"
            f"© 2024 ThoraxScan Team",
        )

    def on_segmentation(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        seg_window = tk.Toplevel(self.root)
        seg_window.title("Segmentación de Imagen")
        seg_window.geometry("300x400")
        seg_window.configure(bg=self.theme["panel_bg"])

        tk.Label(
            seg_window,
            text="Selecciona método de segmentación:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(pady=10)

        methods = [
            ("Umbralización Otsu", "otsu"),
            ("Umbralización Adaptativa", "adaptive"),
            ("Umbral Manual", "threshold"),
            ("Crecimiento de Regiones", "region_growing"),
            ("Watershed", "watershed"),
        ]

        for name, method in methods:
            if method == "threshold":
                cmd = lambda m=method: self.open_manual_threshold_window(seg_window)
            else:
                cmd = lambda m=method: self.apply_segmentation(m, seg_window)

            tk.Button(
                seg_window,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                command=cmd,
            ).pack(fill="x", padx=20, pady=2)

    def on_texture_analysis(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        self.show_message(
            "Análisis de Texturas",
            "Análisis de texturas en desarrollo:\n"
            "- Matriz de co-ocurrencia\n"
            "- Características de Haralick\n"
            "- Filtros Gabor",
        )

    def apply_segmentation(self, method: str, window: tk.Toplevel = None, **params):
        # New flow: accept optional kwargs passed from callers and forward them
        def _log_and_close(win):
            if isinstance(win, tk.Toplevel):
                try:
                    win.destroy()
                except Exception:
                    pass

        try:
            current_image = self.image_controller.get_current_image()
            if not current_image:
                self.show_message("Advertencia", "Primero carga una imagen")
                _log_and_close(window)
                return

            if not hasattr(self.manual_adjustments, "segment_image"):
                logger.error("ManualAdjustments.segment_image no disponible")
                _log_and_close(window)
                return

            # Collect any params passed via closure by callers (they call with kwargs)
            # Python will forward kwargs automatically when callers use keyword args.
            # Here we call segment_image with whatever kwargs were passed to apply_segmentation.
            import inspect

            # Build kwargs passed to this function (exclude method and window)
            frame = inspect.currentframe()
            try:
                # get caller locals/vars is complicated; simpler: rely on callers to pass kwargs explicitly
                pass
            finally:
                del frame

            # Call segment_image forwarding kwargs
            try:
                segmented = self.manual_adjustments.segment_image(
                    current_image, method, **params
                )
            except Exception as e:
                logger.error(f"Error ejecutando segment_image: {e}")
                _log_and_close(window)
                return

            if segmented is None:
                logger.error("segment_image devolvió None")
                _log_and_close(window)
                return

            # Update image manager with the segmented result
            try:
                self.image_controller.image_manager.update_image(
                    segmented, f"Segmentación: {method}"
                )
                self.display_image()
                self.update_histogram()
            except Exception as e:
                logger.error(f"Error actualizando imagen segmentada: {e}")

            _log_and_close(window)
        except Exception as e:
            logger.error(f"Error preparing segmentation: {e}")
        

    def open_adaptive_segmentation_window(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        win = tk.Toplevel(self.root)
        win.title("Umbral Adaptativo")
        win.geometry("360x160")
        win.configure(bg=self.theme["panel_bg"])

        tk.Label(
            win,
            text="Valor Umbral (tamaño de bloque)",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", padx=12, pady=(10, 2))
        # Determine slider max using current image dimensions so block size can span full useful range
        base_image = self.image_controller.get_current_image()
        if base_image:
            img_w, img_h = base_image.size
        else:
            img_w, img_h = 101, 101

        max_block = max(3, min(img_w, img_h))
        # Ensure max_block is odd to make it valid for adaptiveThreshold; if even, reduce by 1
        if max_block % 2 == 0:
            max_block -= 1
        if max_block < 3:
            max_block = 3

        # Start slider at 0 for user clarity; map to valid block when using
        block_var = tk.IntVar(value=0)

        def _on_block_change(v):
            try:
                val = int(float(v))
            except Exception:
                val = block_var.get()

            # enforce odd
            if val < 3:
                # don't change slider UI value, but use 3 for preview/processing
                eff = 3
            else:
                eff = val if val % 2 == 1 else (val + 1 if val + 1 <= max_block else val - 1)
                # if we adjusted to odd, update slider to reflect
                if eff != val:
                    block_var.set(eff)

            # Live preview: apply adaptive segmentation on current image
            try:
                base = self.image_controller.get_current_image()
                if base and hasattr(self.manual_adjustments, "segment_image"):
                    # Create a low-resolution copy for fast preview
                    max_preview_dim = 256
                    bw, bh = base.size
                    scale = min(1.0, max_preview_dim / max(bw, bh))
                    if scale < 1.0:
                        sw = max(3, int(bw * scale))
                        sh = max(3, int(bh * scale))
                        small = base.copy().resize((sw, sh), Image.Resampling.LANCZOS)
                    else:
                        small = base.copy()
                        sw, sh = small.size

                    # Map block size to small image scale to keep behavior similar
                    if bw > 0:
                        mapped = max(3, int(round(eff * (sw / bw))))
                    else:
                        mapped = eff
                    mapped = mapped if mapped % 2 == 1 else mapped + 1
                    if mapped < 3:
                        mapped = 3

                    # Run segmentation on the small image for speed
                    preview_small = self.manual_adjustments.segment_image(
                        small, "adaptive", block_size=int(mapped)
                    )

                    # Upscale preview mask back to full size using nearest neighbor
                    try:
                        upscaled = preview_small.resize((bw, bh), Image.Resampling.NEAREST)
                    except Exception:
                        upscaled = preview_small.resize((bw, bh))

                    self.preview_image = upscaled
                    self.display_image()
            except Exception as e:
                logger.error(f"Error en vista previa adaptativa: {e}")

        # Enforce minimum of 3 for adaptive threshold block size (OpenCV requirement)
        block_scale = tk.Scale(
            win,
            from_=3,
            to=max_block,
            resolution=1,
            orient="horizontal",
            variable=block_var,
            command=_on_block_change,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            length=300,
        )
        block_scale.pack(padx=12)
        # Initialize at minimum valid block size and trigger one preview
        try:
            block_var.set(3)
            block_scale.set(3)
            _on_block_change("3")
        except Exception:
            pass

        def apply_adaptive():
            block = block_var.get()
            try:
                result = self.manual_adjustments.segment_image(
                    self.image_controller.get_current_image(), "adaptive", block_size=int(block)
                )
                # Commit
                self.preview_image = None
                try:
                    self.image_controller.image_manager.update_image(
                        result, f"Umbral adaptativo: {int(block)}"
                    )
                    self.image_controller._emit("image_updated")
                except Exception as e:
                    logger.error(f"Error guardando imagen adaptativa: {e}")
            except Exception as e:
                logger.error(f"Error aplicando umbral adaptativo: {e}")
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        btn_frame = tk.Frame(win, bg=self.theme["panel_bg"])
        btn_frame.pack(fill="x", pady=10)

        tk.Button(
            btn_frame,
            text="Aplicar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=apply_adaptive,
        ).pack(side="left", padx=12)

        def _cancel_adaptive():
            try:
                self.preview_image = None
                self.display_image()
            except Exception as e:
                logger.error(f"Error cancelando vista previa adaptativa: {e}")
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        tk.Button(
            btn_frame,
            text="Cancelar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=_cancel_adaptive,
        ).pack(side="right", padx=12)

    def open_manual_threshold_window(self, parent_window: tk.Toplevel = None):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        win = tk.Toplevel(self.root)
        win.title("Umbral Manual")
        win.geometry("360x140")
        win.configure(bg=self.theme["panel_bg"])

        # Slider label will show actual computed range below (updated after computing max)
        label_text_var = tk.StringVar()
        label_text_var.set("Selecciona el valor de umbral:")
        thr_label = tk.Label(
            win,
            textvariable=label_text_var,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        )
        thr_label.pack(anchor="w", padx=12, pady=(10, 2))

        # Determine appropriate slider max based on image dtype and observed max
        base_img = self.image_controller.get_current_image()
        try:
            img_arr = np.array(base_img)
            # max value possible for the dtype (e.g., uint8 -> 255, uint16 -> 65535)
            max_possible = int(2 ** (img_arr.dtype.itemsize * 8) - 1)
            observed_max = int(img_arr.max()) if img_arr.size > 0 else 255
            # Use observed max so slider reflects real image values; ensure at least 255
            slider_max = max(observed_max, 255)
            # But don't exceed dtype max
            slider_max = min(slider_max, max_possible)
        except Exception:
            slider_max = 255

        # Default threshold: midpoint of observed range (or half of slider_max fallback)
        try:
            default_thr = int(observed_max // 2) if observed_max > 0 else int(slider_max // 2)
            default_thr = max(0, min(default_thr, slider_max))
        except Exception:
            default_thr = min(128, slider_max // 2)

        thr_var = tk.IntVar(value=default_thr)

        def _on_thr_change(v):
            try:
                val = int(float(v))
            except Exception:
                val = thr_var.get()

            # Live preview: apply threshold to a copy of current image and set preview_image
            try:
                base = self.image_controller.get_current_image()
                if base and hasattr(self.manual_adjustments, "segment_image"):
                    # Convert slider value to int and call segmentation
                    preview = self.manual_adjustments.segment_image(
                        base, "threshold", threshold=int(val)
                    )
                    self.preview_image = preview
                    self.display_image()
            except Exception as e:
                logger.error(f"Error en vista previa de umbral: {e}")

        # Update label with actual range
        try:
            label_text_var.set(f"Selecciona el valor de umbral (0 - {slider_max}):")
        except Exception:
            label_text_var.set("Selecciona el valor de umbral:")

        thr_scale = tk.Scale(
            win,
            from_=0,
            to=slider_max,
            resolution=1,
            orient="horizontal",
            variable=thr_var,
            command=_on_thr_change,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            length=320,
        )
        thr_scale.pack(padx=12)

        # Live value label for clarity
        value_var = tk.StringVar()
        value_var.set(f"Valor: {default_thr}")
        val_label = tk.Label(
            win, textvariable=value_var, bg=self.theme["panel_bg"], fg=self.theme["text_color"]
        )
        val_label.pack(anchor="w", padx=12, pady=(6, 0))

        # Ensure the scale shows the default value and trigger preview update
        try:
            thr_scale.config(from_=0, to=slider_max)
            thr_var.set(int(default_thr))
            thr_scale.set(int(default_thr))
            # call preview handler once to show initial preview
            _on_thr_change(str(int(default_thr)))
        except Exception:
            pass

        # Wrap original preview callback to also update the live label
        _orig_on_thr = _on_thr_change

        def _on_thr_change_with_label(v):
            try:
                val = int(float(v))
            except Exception:
                val = thr_var.get()
            try:
                value_var.set(f"Valor: {val}")
            except Exception:
                pass
            _orig_on_thr(v)

        # Rebind the scale command to the wrapper
        try:
            thr_scale.configure(command=_on_thr_change_with_label)
        except Exception:
            pass

        def apply_threshold():
            t = thr_var.get()
            # Commit: clear preview, update manager and emit event
            try:
                result = self.manual_adjustments.segment_image(
                    self.image_controller.get_current_image(), "threshold", threshold=int(t)
                )
                # Commit to image manager
                try:
                    self.preview_image = None
                    self.image_controller.image_manager.update_image(
                        result, f"Umbral manual: {int(t)}"
                    )
                    self.image_controller._emit("image_updated")
                except Exception as e:
                    logger.error(f"Error guardando imagen con umbral: {e}")
            except Exception as e:
                logger.error(f"Error aplicando umbral manual: {e}")
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        btn_frame = tk.Frame(win, bg=self.theme["panel_bg"])
        btn_frame.pack(fill="x", pady=10)

        tk.Button(
            btn_frame,
            text="Aplicar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=apply_threshold,
        ).pack(side="left", padx=12)

        def _cancel():
            # Discard preview and restore displayed image
            try:
                self.preview_image = None
                self.display_image()
            except Exception as e:
                logger.error(f"Error cancelando vista previa de umbral: {e}")
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        tk.Button(
            btn_frame,
            text="Cancelar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=_cancel,
        ).pack(side="right", padx=12)

    def on_cluster_segmentation(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        k = simpledialog.askinteger(
            "k-means", "Número de clusters (k):", parent=self.root, minvalue=2, maxvalue=20
        )
        if k is None:
            return
        # store the source image and chosen k so stats can be computed later without re-prompt
        try:
            base = self.image_controller.get_current_image()
            self._last_kmeans_source = base.copy() if base else None
            self._last_kmeans_k = int(k)
        except Exception:
            self._last_kmeans_source = None
            self._last_kmeans_k = int(k)

        # Run kmeans in a background thread and show progress dialog
        progress_win = tk.Toplevel(self.root)
        progress_win.title("K-means: procesando...")
        progress_win.geometry("360x80")
        progress_win.configure(bg=self.theme["panel_bg"])
        progress_win.transient(self.root)
        progress_win.grab_set()

        tk.Label(
            progress_win,
            text=f"Ejecutando k-means (k={k})...",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", padx=12, pady=(8, 4))

        pb = ttk.Progressbar(progress_win, mode="indeterminate")
        pb.pack(fill="x", padx=12, pady=(0, 8))
        pb.start(50)

        def _run_kmeans():
            try:
                base = self.image_controller.get_current_image()
                if base is None:
                    raise RuntimeError("No hay imagen para segmentar")

                # compute segmentation (this may take time)
                result = self.manual_adjustments.segment_image(base, "kmeans", k=int(k))

                def _on_done():
                    try:
                        pb.stop()
                        progress_win.grab_release()
                        progress_win.destroy()
                    except Exception:
                        pass

                    try:
                        self.image_controller.image_manager.update_image(
                            result, f"K-means (k={k})"
                        )
                        self.image_controller._emit("image_updated")
                    except Exception as e:
                        logger.error(f"Error guardando resultado k-means: {e}")
                        self.show_message("Error", f"No se pudo actualizar imagen: {e}")

                # schedule UI update
                self.root.after(50, _on_done)

            except Exception as e:
                logger.error(f"Error en k-means background: {e}")

                def _on_error():
                    try:
                        pb.stop()
                        progress_win.grab_release()
                        progress_win.destroy()
                    except Exception:
                        pass
                    self.show_message("Error", f"K-means falló: {e}")

                self.root.after(50, _on_error)

        thread = threading.Thread(target=_run_kmeans, daemon=True)
        thread.start()

    def on_cluster_stats(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return
        # Prefer using the last performed k-means (no prompt). If not available, ask for k.
        k = getattr(self, "_last_kmeans_k", None)
        source = getattr(self, "_last_kmeans_source", None)
        if k is None or source is None:
            k = simpledialog.askinteger(
                "k-means Stats", "Número de clusters (k):", parent=self.root, minvalue=1, maxvalue=50
            )
            if k is None:
                return
            base = self.image_controller.get_current_image()
        else:
            base = source

        try:
            results = self.manual_adjustments.kmeans_cluster_stats(base, k=int(k))

            st_win = tk.Toplevel(self.root)
            st_win.title(f"Área y Perímetro por cluster (k={k})")
            st_win.geometry("420x320")

            txt = tk.Text(st_win, wrap="word")
            txt.pack(fill="both", expand=True)
            for r in results:
                txt.insert(
                    "end",
                    f"Cluster {r['label']}: Área(píxeles)={r['area']}, Perímetro(píxeles)={r['perimeter']:.2f}, Count={r['count']}\n",
                )
            txt.config(state="disabled")

        except Exception as e:
            logger.error(f"Error calculando stats por cluster: {e}")
            self.show_message("Error", f"No se pudieron calcular estadísticas: {e}")

    def on_current_threshold_stats(self):
        """Compute area & perimeter for the currently displayed image (useful after applying a threshold)."""
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        try:
            img = self.image_controller.get_current_image()
            if img is None:
                self.show_message("Advertencia", "No hay imagen disponible")
                return

            stats = self.manual_adjustments.compute_mask_stats(img)

            st_win = tk.Toplevel(self.root)
            st_win.title("Área y Perímetro - Imagen actual")
            st_win.geometry("420x320")

            txt = tk.Text(st_win, wrap="word")
            txt.pack(fill="both", expand=True)
            if not stats:
                txt.insert("end", "No se encontraron regiones o no es una máscara binaria\n")
            else:
                for label, vals in stats.items():
                    txt.insert("end", f"Label: {label} → Área(píxeles): {vals['area']}, Perímetro(píxeles): {vals['perimeter']:.2f}\n")
            txt.config(state="disabled")

        except Exception as e:
            logger.error(f"Error calculando estadísticas de imagen actual: {e}")
            self.show_message("Error", f"No se pudieron calcular estadísticas: {e}")

    def open_binary_threshold_window(self):
        """Open a simple binary threshold dialog (no live preview)."""
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        win = tk.Toplevel(self.root)
        win.title("Umbral Binario")
        win.geometry("360x140")
        win.configure(bg=self.theme["panel_bg"])

        tk.Label(
            win,
            text="Selecciona el valor de umbral para binarización:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", padx=12, pady=(10, 2))

        # Determine slider range from image dtype/observed values
        base_img = self.image_controller.get_current_image()
        try:
            img_arr = np.array(base_img)
            max_possible = int(2 ** (img_arr.dtype.itemsize * 8) - 1)
            observed_max = int(img_arr.max()) if img_arr.size > 0 else 255
            slider_max = max(observed_max, 255)
            slider_max = min(slider_max, max_possible)
        except Exception:
            slider_max = 255

        default_thr = int(observed_max // 2) if 'observed_max' in locals() else int(slider_max // 2)
        thr_var = tk.IntVar(value=default_thr)

        thr_scale = tk.Scale(
            win,
            from_=0,
            to=slider_max,
            resolution=1,
            orient="horizontal",
            variable=thr_var,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            length=320,
        )
        thr_scale.pack(padx=12)

        # Full-resolution live preview for binary threshold (may be slower on large images)
        def _on_thr_change_fullres(v):
            try:
                val = int(float(v))
            except Exception:
                val = thr_var.get()

            try:
                base = self.image_controller.get_current_image()
                if base and hasattr(self.manual_adjustments, "segment_image"):
                    preview = self.manual_adjustments.segment_image(
                        base, "threshold", threshold=int(val)
                    )
                    self.preview_image = preview
                    self.display_image()
            except Exception as e:
                logger.error(f"Error en vista previa umbral binario (full-res): {e}")

        # Bind preview callback and trigger initial preview
        try:
            thr_scale.configure(command=_on_thr_change_fullres)
            thr_scale.set(int(thr_var.get()))
            _on_thr_change_fullres(str(int(thr_var.get())))
        except Exception:
            pass

        def apply_binary():
            t = int(thr_var.get())
            try:
                result = self.manual_adjustments.segment_image(
                    self.image_controller.get_current_image(), "threshold", threshold=t
                )
                # Commit result
                try:
                    self.preview_image = None
                    self.image_controller.image_manager.update_image(
                        result, f"Umbral binario: {t}"
                    )
                    self.image_controller._emit("image_updated")
                except Exception as e:
                    logger.error(f"Error guardando imagen con umbral binario: {e}")
                    self.show_message("Error", f"No se pudo actualizar imagen: {e}")
            except Exception as e:
                logger.error(f"Error aplicando umbral binario: {e}")
                self.show_message("Error", f"Fallo al aplicar umbral: {e}")
            finally:
                try:
                    win.destroy()
                except Exception:
                    pass

        def _show_stats_for_threshold(t: int):
            try:
                seg = self.manual_adjustments.segment_image(
                    self.image_controller.get_current_image(), "threshold", threshold=int(t)
                )
                stats = self.manual_adjustments.compute_mask_stats(seg)
                # show results in dialog
                st_win = tk.Toplevel(self.root)
                st_win.title(f"Área y Perímetro - Umbral {t}")
                st_win.geometry("400x300")
                txt = tk.Text(st_win, wrap="word")
                txt.pack(fill="both", expand=True)
                for label, vals in stats.items():
                    txt.insert("end", f"Label: {label} → Área(píxeles): {vals['area']}, Perímetro(píxeles): {vals['perimeter']:.2f}\n")
                txt.config(state="disabled")
            except Exception as e:
                logger.error(f"Error mostrando stats umbral: {e}")
                self.show_message("Error", f"No se pudieron calcular estadísticas: {e}")

        btn_frame = tk.Frame(win, bg=self.theme["panel_bg"])
        btn_frame.pack(fill="x", pady=10)

        tk.Button(
            btn_frame,
            text="Aplicar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=apply_binary,
        ).pack(side="left", padx=12)

        tk.Button(
            btn_frame,
            text="Área y Perímetro",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=lambda: _show_stats_for_threshold(int(thr_var.get())),
        ).pack(side="left", padx=6)

        def _cancel_bin():
            try:
                self.preview_image = None
                self.display_image()
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass

        tk.Button(
            btn_frame,
            text="Cancelar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=_cancel_bin,
        ).pack(side="right", padx=12)

    def update_histogram(self):
        hist_data = self.image_controller.get_histogram_data()
        if hist_data and self.histogram_panel:
            self.histogram_panel.update_histogram(hist_data)

    def on_show_histogram(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        hist_window = tk.Toplevel(self.root)
        hist_window.title("Histograma Detallado")
        hist_window.geometry("800x600")
        hist_window.configure(bg=self.theme["panel_bg"])

        expanded_histogram = HistogramPanel(hist_window, self.theme, self.fonts)

        hist_data = self.image_controller.get_histogram_data()
        if hist_data:
            expanded_histogram.update_histogram(hist_data)

    def on_toggle_theme(self):
        current = self.styles.get("current_theme", "dark")
        themes = list(self.styles["themes"].keys())

        next_index = (themes.index(current) + 1) % len(themes)
        next_theme = themes[next_index]

        self.styles["current_theme"] = next_theme
        self.save_styles()

        self.load_styles()
        self.apply_global_style()
        self.apply_widgets_style()

        self.show_message("Tema cambiado", f"Has cambiado al tema: {next_theme}")

    def save_styles(self):
        with open(STYLES_PATH, "w", encoding="utf-8") as f:
            json.dump(self.styles, f, indent=4, ensure_ascii=False)

    def apply_widgets_style(self):
        def update(widget):
            cls = widget.__class__.__name__

            if cls in ("Button", "tk.Button"):
                try:
                    widget.configure(
                        bg=self.theme["button_bg"],
                        fg=self.theme["button_fg"],
                        activebackground=self.theme["button_hover"],
                    )
                except:
                    pass

            elif cls == "Menu":
                try:
                    widget.configure(bg=self.theme["menu_bg"], fg=self.theme["menu_fg"])
                except:
                    pass

            elif cls in ("Frame", "LabelFrame"):
                try:
                    widget.configure(bg=self.theme["panel_bg"])
                except:
                    pass

            else:
                try:
                    widget.configure(
                        bg=self.theme["global_bg"], fg=self.theme["text_color"]
                    )
                except:
                    pass

            for child in widget.winfo_children():
                update(child)

        update(self.root)

    def on_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

        self.clear_mouse_guides()

        if self.image_controller.has_image():
            current_image = self.image_controller.get_current_image()
            if current_image:
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img_width, img_height = current_image.size

                scale_x = img_width / canvas_width if canvas_width > 0 else 1
                scale_y = img_height / canvas_height if canvas_height > 0 else 1

                img_x = int(event.x * scale_x)
                img_y = int(event.y * scale_y)

                pixel_value = "N/A"
                if 0 <= img_x < img_width and 0 <= img_y < img_height:
                    img_array = np.array(current_image)
                    if len(img_array.shape) == 2:  # Grayscale
                        pixel_value = img_array[img_y, img_x]
                    else:  # RGB
                        pixel_value = img_array[img_y, img_x]

                coord_info = f"Pos: ({img_x}, {img_y}) | Pixel: {pixel_value}"
                self.status_label.config(text=coord_info)

                self.draw_mouse_guides(event.x, event.y, canvas_width, canvas_height)

    def clear_mouse_guides(self):
        if hasattr(self, "guide_lines"):
            for line in self.guide_lines:
                self.canvas.delete(line)
            self.guide_lines = []

    def draw_mouse_guides(self, x, y, canvas_width, canvas_height):
        self.clear_mouse_guides()

        line_h = self.canvas.create_line(
            0, y, canvas_width, y, fill="#FF4444", width=1, dash=(4, 2)
        )
        line_v = self.canvas.create_line(
            x, 0, x, canvas_height, fill="#4444FF", width=1, dash=(4, 2)
        )

        self.guide_lines.extend([line_h, line_v])

        dot = self.canvas.create_oval(
            x - 3, y - 3, x + 3, y + 3, fill="#FFFF00", outline="#FF0000", width=2
        )
        self.guide_lines.append(dot)

        if self.image_controller.has_image():
            current_image = self.image_controller.get_current_image()
            if current_image:
                img_width, img_height = current_image.size
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                scale_x = img_width / canvas_width if canvas_width > 0 else 1
                scale_y = img_height / canvas_height if canvas_height > 0 else 1
                img_x = int(x * scale_x)
                img_y = int(y * scale_y)

                coord_x = self.canvas.create_text(
                    canvas_width - 50,
                    y,
                    text=f"Y: {img_y}",
                    fill="#FF4444",
                    font=("Arial", 10, "bold"),
                    anchor="e",
                )
                coord_y = self.canvas.create_text(
                    x,
                    15,
                    text=f"X: {img_x}",
                    fill="#4444FF",
                    font=("Arial", 10, "bold"),
                    anchor="n",
                )

                coord_center = self.canvas.create_text(
                    x + 15,
                    y - 15,
                    text=f"({img_x}, {img_y})",
                    fill="#FFFFFF",
                    font=("Arial", 9, "bold"),
                    anchor="nw",
                )

                self.guide_lines.extend([coord_x, coord_y, coord_center])

    def on_mouse_leave(self, event):
        self.status_label.config(text="Listo")
        self.clear_mouse_guides()

    def on_adjust_brightness(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        adjustments_window = tk.Toplevel(self.root)
        adjustments_window.title("Ajustes Avanzados de Brillo y Contraste")
        adjustments_window.geometry("400x200")
        adjustments_window.configure(bg=self.theme["panel_bg"])

        main_frame = tk.Frame(
            adjustments_window, bg=self.theme["panel_bg"], padx=20, pady=20
        )
        main_frame.pack(fill="both", expand=True)

        brightness_var = tk.DoubleVar(value=1.0)
        contrast_var = tk.DoubleVar(value=1.0)

        tk.Label(
            main_frame,
            text="Brillo:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w")

        brightness_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        brightness_frame.pack(fill="x", pady=5)

        brightness_scale = tk.Scale(
            brightness_frame,
            from_=0.1,
            to=3.0,
            resolution=0.01,
            variable=brightness_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
            length=250,
            command=lambda v: self.apply_advanced_adjustments(
                brightness_var.get(), contrast_var.get()
            ),
        )
        brightness_scale.pack(side="left")

        tk.Label(
            main_frame,
            text="Contraste:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", pady=(10, 0))

        contrast_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        contrast_frame.pack(fill="x", pady=5)

        contrast_scale = tk.Scale(
            contrast_frame,
            from_=0.1,
            to=3.0,
            resolution=0.01,
            variable=contrast_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
            length=250,
            command=lambda v: self.apply_advanced_adjustments(
                brightness_var.get(), contrast_var.get()
            ),
        )
        contrast_scale.pack(side="left")

        button_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        button_frame.pack(fill="x", pady=20)

        tk.Button(
            button_frame,
            text="Aplicar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=lambda: self.apply_advanced_adjustments(
                brightness_var.get(), contrast_var.get()
            ),
        ).pack(side="left", padx=5)

        tk.Button(
            button_frame,
            text="Cerrar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=adjustments_window.destroy,
        ).pack(side="right", padx=5)

    def apply_advanced_adjustments(self, brightness: float, contrast: float):
        self.image_controller.apply_brightness_contrast(brightness, contrast)

    def on_save_image(self):
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        if (
            hasattr(self.image_controller.image_manager, "image_path")
            and self.image_controller.image_manager.image_path
            and self.image_controller.image_manager.image_path.endswith(".dcm")
        ):
            self.on_export_image()
        else:
            try:
                if (
                    hasattr(self.image_controller.image_manager, "image_path")
                    and self.image_controller.image_manager.image_path
                ):
                    # Sobrescribir archivo original
                    self.image_controller.image_manager.current_image.save(
                        self.image_controller.image_manager.image_path
                    )
                    self.show_message("Éxito", "Imagen guardada correctamente")
                else:
                    # Si no hay ruta original, usar exportar
                    self.on_export_image()
            except Exception as e:
                self.show_message("Error", f"No se pudo guardar: {str(e)}")

    def on_filters_menu(self):
        """Menú de filtros avanzados"""
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        filters_window = tk.Toplevel(self.root)
        filters_window.title("Filtros Avanzados")
        filters_window.geometry("300x400")
        filters_window.configure(bg=self.theme["panel_bg"])

        tk.Label(
            filters_window,
            text="Selecciona un filtro:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(pady=10)

        filters = [
            ("Enfoque", "sharpen"),
            ("Desenfoque", "blur"),
            ("Realce de Bordes", "edge_enhance"),
            ("Suavizado", "smooth"),
            ("Detalle", "detail"),
            ("Relieve", "emboss"),
            ("Contorno", "contour"),
            ("Detección de Bordes", "find_edges"),
            ("Desenfoque Gaussiano", "gaussian_blur"),
            ("Máscara de Enfoque", "unsharp_mask"),
        ]

        for name, filter_type in filters:
            tk.Button(
                filters_window,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                command=lambda ft=filter_type: self.apply_advanced_filter(
                    ft, filters_window
                ),
            ).pack(fill="x", padx=20, pady=2)

    def apply_advanced_filter(self, filter_type: str, window: tk.Toplevel):
        """Aplica un filtro avanzado y cierra la ventana"""
        self.image_controller.apply_filter(filter_type)
        window.destroy()

    def get_normalization_description(self, method: str) -> str:
        descriptions = {
            "clahe": "CLAHE (OpenCV) - Mejor para imágenes médicas, preserva detalles locales",
            "histogram": "Ecualización de Histograma (OpenCV) - Mejora contraste global",
            "global": "Normalización Min-Max (OpenCV) - Ajusta todo el rango dinámico",
            "contrast_stretch": "Estiramiento de Contraste (OpenCV) - Usa percentiles 2-98%",
            "local": "Normalización Local (OpenCV) - Adaptativa por regiones usando Gaussianas",
            "percentile": "Normalización por Percentiles (OpenCV) - Usa percentiles 5-95% para rayos X",
        }
        return descriptions.get(method, "Método de normalización OpenCV")

    def show_welcome_message(self):
        WelcomePopup(self.root, self.theme, self.fonts)

    # ============ MÉTODOS PLACEHOLDER ============

    def on_config(self):
        self.show_message(
            "Info", "Menu de Configuración, estara disponible en proximas versiones."
        )

    def on_levels_adjust(self):
        self.show_message("Info", "Ajuste de niveles en desarrollo")
