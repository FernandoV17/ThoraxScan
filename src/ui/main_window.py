import json
import os
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from src.helpers.logger import get_module_logger
from src.processing.auto_enhancer import AutoEnhancer
from src.processing.manual_adjustments import ManualAdjustments
from src.ui.controllers.analysis_controller import AnalysisController
from src.ui.controllers.image_controller import ImageController
from src.ui.controllers.version_controller import VersionController
from src.ui.panels.adjustments_panel import AdjustmentsPanel
from src.ui.panels.filters_panel import FiltersPanel
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

        self.image_controller.set_manual_adjustments(self.manual_adjustments)
        self.image_controller.set_analysis_controller(self.analysis_controller)

        self.setup_ui()
        logger.info("MainWindow inicializado correctamente")

    def setup_ui(self):
        self.root.title(self.version.get_full_version())
        self.root.state('zoomed')

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
            ("Detección de Anomalías", self.on_detect_anomalies),
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
            self.center_panel, bg=self.theme["panel_bg"], height=120
        )
        self.hist_frame.pack(side="bottom", fill="x", pady=(5, 0))
        self.hist_frame.pack_propagate(False)

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

    def on_detect_anomalies(self):
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
        menu.add_command(
            label="Detección de anomalías", command=self.on_detect_anomalies
        )
        menu.add_command(
            label="Detección de fracturas", command=self.on_detect_fractures
        )
        menu.add_command(label="Cardiomegalia", command=self.on_detect_heart)
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

    # ============ MÉTODOS PLACEHOLDER ============

    def on_save_image(self):
        self.show_message("Info", "Función de guardar en desarrollo")

    def on_adjust_brightness(self):
        self.show_message("Info", "Ajustes avanzados en desarrollo")

    def on_filters_menu(self):
        self.show_message("Info", "Menú de filtros en desarrollo")

    def on_show_histogram(self):
        """Muestra histograma detallado"""
        self.show_message("Info", "Histograma detallado en desarrollo")

    def on_toggle_theme(self):
        self.show_message("Info", "Cambio de tema en desarrollo")

    def on_config(self):
        self.show_message("Info", "Configuración en desarrollo")

    def on_levels_adjust(self):
        self.show_message("Info", "Ajuste de niveles en desarrollo")

    def show_welcome_message(self):
        WelcomePopup(self.root, self.theme, self.fonts)
