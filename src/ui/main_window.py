import json
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
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

        analysis_buttons = [
            ("Segmentacion", self.Segmentacion),
            ("Detección de Fracturas", self.on_detect_fractures),
            ("Cardiomegalia", self.on_detect_heart),
        ]

        for text, command in analysis_buttons:
            btn = tk.Button(
                self.right_panel,
                text=text,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                bd=0,
                padx=10,
                pady=10,
                font=(self.fonts["main"]["family"], 10),
                cursor="hand2",
                command=command,
            )
            btn.pack(fill="x", pady=4)

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
            tk.Button(
                seg_window,
                text=name,
                bg=self.theme["accent"],
                fg=self.theme["text_color"],
                command=lambda m=method: self.apply_segmentation(m, seg_window),
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

    def apply_segmentation(self, method: str, window: tk.Toplevel):
        try:
            current_image = self.image_controller.get_current_image()
            if current_image and hasattr(self.manual_adjustments, "segment_image"):
                segmented = self.manual_adjustments.segment_image(current_image, method)
                self.image_controller.image_manager.update_image(
                    segmented, f"Segmentación: {method}"
                )
                self.display_image()
                self.update_histogram()
                window.destroy()
        except Exception as e:
            logger.error(f"Error en segmentación: {e}")

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

    def on_fft_analysis(self):
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            result = self.analysis_controller.analyze_frequency_domain(image)
            self.show_analysis_result("Análisis FFT - Dominio de Frecuencia", result)
        else:
            self.show_message("Advertencia", "Primero carga una imagen")

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
        menu.add_command(
            label="Detección de anomalías", command=self.on_detect_anomalies
        )
        menu.add_command(
            label="Detección de fracturas", command=self.on_detect_fractures
        )
        menu.add_command(label="Cardiomegalia", command=self.on_detect_heart)
        menu.add_command(
            label="Análisis FFT (Frecuencia)",
            command=self.on_fft_analysis,
        )
        menu.add_command(label="Segmentación", command=self.on_segmentation)
        menu.add_command(label="Análisis de texturas", command=self.on_texture_analysis)
        menu.add_separator()
        menu.add_command(label="Mostrar histograma", command=self.on_show_histogram)

        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def on_settings(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Tema claro/oscuro", command=self.on_toggle_theme)
        menu.add_command(label="Configuración", command=self.on_config)
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())


def on_apply_filter(self, filter_type: str, params=None):
    if params is None:
        params = self.get_filter_parameters(filter_type)
    self.image_controller.apply_filter(filter_type, params)

    def get_filter_parameters(self, filter_type):
        param_window = tk.Toplevel(self.root)
        param_window.title(f"Parámetros - {filter_type}")
        param_window.geometry("300x200")

        params = {}

        if filter_type == "brightness_contrast":
            brightness_var = tk.DoubleVar(value=1.0)
            contrast_var = tk.DoubleVar(value=1.0)

            tk.Label(param_window, text="Brillo:").pack()
            tk.Scale(
                param_window,
                from_=0.1,
                to=3.0,
                resolution=0.1,
                variable=brightness_var,
                orient="horizontal",
            ).pack()

            tk.Label(param_window, text="Contraste:").pack()
            tk.Scale(
                param_window,
                from_=0.1,
                to=3.0,
                resolution=0.1,
                variable=contrast_var,
                orient="horizontal",
            ).pack()

            def apply_params():
                params["brightness"] = brightness_var.get()
                params["contrast"] = contrast_var.get()
                param_window.destroy()
                self.image_controller.apply_filter(filter_type, params)

            tk.Button(param_window, text="Aplicar", command=apply_params).pack()

        return params

    def show_welcome_message(self):
        WelcomePopup(self.root, self.theme, self.fonts)

    # ============ MÉTODOS PLACEHOLDER ============

    def on_config(self):
        self.show_message(
            "Info", "Menu de Configuración, estara disponible en proximas versiones."
        )

    def on_levels_adjust(self):
        self.show_message("Info", "Ajuste de niveles en desarrollo")

    def on_fft_analysis(self):
        """Interfaz completa de análisis Fourier"""
        if not self.image_controller.has_image():
            self.show_message("Advertencia", "Primero carga una imagen")
            return

        fft_window = tk.Toplevel(self.root)
        fft_window.title("Análisis Fourier - Dominio de Frecuencia")
        fft_window.geometry("500x600")
        fft_window.configure(bg=self.theme["panel_bg"])

        main_frame = tk.Frame(fft_window, bg=self.theme["panel_bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        # Título
        tk.Label(
            main_frame,
            text="Análisis en Dominio de Frecuencia",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], 14, "bold"),
        ).pack(anchor="w", pady=(0, 20))

        # Sección de análisis
        analysis_frame = tk.LabelFrame(
            main_frame,
            text=" Análisis Espectral ",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        )
        analysis_frame.pack(fill="x", pady=(0, 15))

        tk.Button(
            analysis_frame,
            text="🔍 Analizar Espectro FFT",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10),
            command=self._run_fft_analysis,
            width=20,
        ).pack(pady=10)

        # Sección de filtros de frecuencia
        filter_frame = tk.LabelFrame(
            main_frame,
            text=" Filtros de Frecuencia ",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        )
        filter_frame.pack(fill="x", pady=(0, 15))

        # Tipo de filtro
        tk.Label(
            filter_frame,
            text="Tipo de Filtro:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(anchor="w", pady=(5, 0))

        filter_type_var = tk.StringVar(value="high")
        filter_types = [
            ("Pasa Altas (Enfocar)", "high"),
            ("Pasa Bajas (Suavizar)", "low"),
            ("Pasa Banda", "band"),
            ("Rechaza Banda", "band_stop"),
        ]

        filter_type_frame = tk.Frame(filter_frame, bg=self.theme["panel_bg"])
        filter_type_frame.pack(fill="x", pady=5)

        for text, value in filter_types:
            tk.Radiobutton(
                filter_type_frame,
                text=text,
                variable=filter_type_var,
                value=value,
                bg=self.theme["panel_bg"],
                fg=self.theme["text_color"],
                selectcolor=self.theme["accent"],
            ).pack(side="left", padx=10)

        # Parámetros del filtro
        params_frame = tk.Frame(filter_frame, bg=self.theme["panel_bg"])
        params_frame.pack(fill="x", pady=10)

        # Cutoff frequency
        tk.Label(
            params_frame,
            text="Frecuencia de Corte:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).grid(row=0, column=0, sticky="w", pady=5)

        cutoff_var = tk.IntVar(value=30)
        cutoff_scale = tk.Scale(
            params_frame,
            from_=10,
            to=100,
            variable=cutoff_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
            length=200,
        )
        cutoff_scale.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        # Strength
        tk.Label(
            params_frame,
            text="Fuerza del Filtro:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).grid(row=1, column=0, sticky="w", pady=5)

        strength_var = tk.DoubleVar(value=1.0)
        strength_scale = tk.Scale(
            params_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            variable=strength_var,
            orient="horizontal",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            troughcolor=self.theme["accent"],
            length=200,
        )
        strength_scale.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        params_frame.columnconfigure(1, weight=1)

        # Botones de acción
        button_frame = tk.Frame(main_frame, bg=self.theme["panel_bg"])
        button_frame.pack(fill="x", pady=20)

        def apply_frequency_filter():
            try:
                current_image = self.image_controller.get_current_image()
                filtered_image = self.analysis_controller.apply_frequency_filter(
                    current_image,
                    filter_type_var.get(),
                    cutoff_var.get(),
                    strength_var.get(),
                )

                self.image_controller.image_manager.update_image(
                    filtered_image, f"Filtro Frecuencia: {filter_type_var.get()}"
                )
                self.display_image()
                self.update_histogram()

                self.show_message(
                    "Éxito",
                    f"Filtro {filter_type_var.get()} aplicado correctamente\n"
                    f"Cutoff: {cutoff_var.get()}, Fuerza: {strength_var.get():.1f}",
                )

            except Exception as e:
                logger.error(f"Error aplicando filtro frecuencia: {e}")
                self.show_message("Error", f"Error aplicando filtro: {str(e)}")

        def show_spectrum():
            try:
                current_image = self.image_controller.get_current_image()
                spectrum_image = self.analysis_controller.get_frequency_spectrum(
                    current_image
                )

                # Mostrar en ventana separada
                spectrum_window = tk.Toplevel(self.root)
                spectrum_window.title("Espectro de Frecuencia")
                spectrum_window.geometry("400x400")

                # Convertir para mostrar en tkinter
                from PIL import ImageTk

                spectrum_photo = ImageTk.PhotoImage(spectrum_image)

                spectrum_label = tk.Label(spectrum_window, image=spectrum_photo)
                spectrum_label.image = spectrum_photo  # Mantener referencia
                spectrum_label.pack(padx=10, pady=10)

                tk.Label(
                    spectrum_window,
                    text="Espectro de Frecuencia (FFT Magnitude)",
                    font=("Arial", 10, "bold"),
                ).pack(pady=(0, 10))

            except Exception as e:
                logger.error(f"Error mostrando espectro: {e}")
                self.show_message("Error", f"Error mostrando espectro: {str(e)}")

        tk.Button(
            button_frame,
            text="Aplicar Filtro",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
            command=apply_frequency_filter,
        ).pack(side="left", padx=5)

        tk.Button(
            button_frame,
            text="Ver Espectro",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=show_spectrum,
        ).pack(side="left", padx=5)

        tk.Button(
            button_frame,
            text="Cerrar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=fft_window.destroy,
        ).pack(side="right", padx=5)

    def _run_fft_analysis(self):
        """Ejecuta el análisis FFT y muestra resultados"""
        if self.image_controller.has_image():
            image = self.image_controller.get_current_image()
            result = self.analysis_controller.analyze_frequency_domain(image)
            self.show_analysis_result("Análisis FFT - Dominio de Frecuencia", result)
