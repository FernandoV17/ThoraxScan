import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from src.analysis.controller import AnalysisController
from src.core.image_manager import ImageManager
from src.core.metadata_extractor import MetadataExtractor
from src.processing.auto_enhancer import AutoEnhancer
from src.ui.controllers.version_controller import VersionController
from src.ui.widgets.welcome_popup import WelcomePopup

STYLES_PATH = os.path.join("assets", "ui", "styles.json")
LOGO_PATH = os.path.join("assets", "icons", "logo.png")


class MainWindow:
    def __init__(self, root):
        self.root = root  # ← DEBE IR PRIMERO
        self.version = VersionController()

        # Inicializar managers
        self.image_manager = ImageManager()
        self.metadata_extractor = MetadataExtractor()
        self.auto_enhancer = AutoEnhancer()
        self.analysis_controller = AnalysisController()

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
            logo_img = Image.open(LOGO_PATH).resize((100, 100))
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

    def on_file(self):
        """Menú Archivo"""
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
        menu.add_command(label="Filtros", command=self.on_filters)
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

    def on_open_image(self):
        file_types = [
            ("Imágenes médicas", "*.dcm"),
            ("Imágenes", "*.png *.jpg *.jpeg *.tiff *.bmp"),
            ("Todos los archivos", "*.*"),
        ]

        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de rayos X", filetypes=file_types
        )

        if file_path:
            try:
                success, message = self.image_manager.load_image(file_path)
                if success:
                    self.display_image()
                    self.update_metadata()
                    self.update_histogram()
                else:
                    messagebox.showerror(
                        "Error", f"No se pudo cargar la imagen: {message}"
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")

    def display_image(self):
        if self.image_manager.current_image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img = self.image_manager.current_image.copy()
                img.thumbnail(
                    (canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS
                )

                self.tk_image = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=self.tk_image,
                    anchor="center",
                )

    def update_metadata(self):
        if self.image_manager.current_image:
            metadata = self.metadata_extractor.extract_metadata(self.image_manager)

            self.meta_box.config(state="normal")
            self.meta_box.delete("1.0", "end")

            for key, value in metadata.items():
                self.meta_box.insert("end", f"{key}: {value}\n")

            self.meta_box.config(state="disabled")

    def update_histogram(self):
        # Aquí integrarías con src.ui.histogram_panel
        pass

    def on_auto_enhance(self):
        if self.image_manager.current_image:
            try:
                enhanced_image = self.auto_enhancer.enhance(
                    self.image_manager.current_image
                )
                self.image_manager.current_image = enhanced_image
                self.display_image()
                messagebox.showinfo("Éxito", "Imagen mejorada automáticamente")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo mejorar la imagen: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")

    def on_restore(self):
        if self.image_manager.original_image:
            self.image_manager.restore_original()
            self.display_image()
            messagebox.showinfo("Éxito", "Imagen restaurada a su estado original")
        else:
            messagebox.showwarning(
                "Advertencia", "No hay imagen original para restaurar"
            )

    def on_detect_anomalies(self):
        if self.image_manager.current_image:
            try:
                result = self.analysis_controller.detect_anomalies(
                    self.image_manager.current_image
                )
                self.show_analysis_result("Anomalías Detectadas", result)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error en detección de anomalías: {str(e)}"
                )
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")

    def on_detect_fractures(self):
        if self.image_manager.current_image:
            try:
                result = self.analysis_controller.detect_fractures(
                    self.image_manager.current_image
                )
                self.show_analysis_result("Fracturas Detectadas", result)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error en detección de fracturas: {str(e)}"
                )
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")

    def on_detect_heart(self):
        if self.image_manager.current_image:
            try:
                result = self.analysis_controller.detect_cardiomegaly(
                    self.image_manager.current_image
                )
                self.show_analysis_result("Cardiomegalia", result)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error en análisis de cardiomegalia: {str(e)}"
                )
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")

    def show_analysis_result(self, title, result):
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.geometry("400x300")

        text = tk.Text(result_window, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", str(result))
        text.config(state="disabled")

    def on_save_image(self):
        if self.image_manager.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tiff")],
            )
            if file_path:
                self.image_manager.current_image.save(file_path)
                messagebox.showinfo("Éxito", "Imagen guardada correctamente")

    def on_export_image(self):
        # Implementar exportación con metadatos embebidos
        pass

    def on_adjust_brightness(self):
        # Integrar con src.processing.manual_adjustments
        pass

    def on_filters(self):
        # Integrar con src.processing.edge_detection y otros
        pass

    def on_show_histogram(self):
        # Integrar con src.ui.histogram_panel
        pass

    def on_toggle_theme(self):
        # Implementar cambio de tema
        pass

    def on_config(self):
        # Implementar ventana de configuración
        pass

    def show_welcome_message(self):
        WelcomePopup(self.root, self.theme, self.fonts)
