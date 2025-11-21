import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class HistogramPanel:
    def __init__(self, parent, theme: Dict[str, Any], fonts: Dict[str, Any]):
        self.parent = parent
        self.theme = theme
        self.fonts = fonts
        self.current_histogram_data = None
        self.figure = None
        self.canvas = None
        self.ax = None

        self.create_widgets()
        logger.info("HistogramPanel inicializado")

    def create_widgets(self):
        """Crea la interfaz completa del panel de histograma"""
        # Frame principal
        self.main_frame = tk.Frame(self.parent, bg=self.theme["panel_bg"])
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Título y controles
        self.create_header()

        # Gráfico del histograma
        self.create_histogram_plot()

        # Estadísticas
        self.create_stats_section()

    def create_header(self):
        """Crea el encabezado con controles"""
        header_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        header_frame.pack(fill="x", pady=(0, 10))

        # Título
        title_label = tk.Label(
            header_frame,
            text="📊 Histograma en Tiempo Real",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        )
        title_label.pack(side="left")

        # Controles a la derecha
        controls_frame = tk.Frame(header_frame, bg=self.theme["panel_bg"])
        controls_frame.pack(side="right")

        # Botón de exportar
        export_btn = tk.Button(
            controls_frame,
            text="Exportar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 8),
            command=self.export_histogram,
        )
        export_btn.pack(side="left", padx=2)

        # Botón de estadísticas
        stats_btn = tk.Button(
            controls_frame,
            text="Stats",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 8),
            command=self.show_detailed_statistics,
        )
        stats_btn.pack(side="left", padx=2)

    def create_histogram_plot(self):
        """Crea el gráfico del histograma con matplotlib"""
        # Frame para el gráfico
        plot_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        plot_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Crear figura de matplotlib
        self.figure = Figure(figsize=(6, 3), dpi=80, facecolor=self.theme["panel_bg"])
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.theme["panel_bg"])

        # Configurar colores del gráfico
        self.ax.tick_params(colors=self.theme["text_color"], labelsize=8)
        self.ax.spines["bottom"].set_color(self.theme["text_color"])
        self.ax.spines["top"].set_color(self.theme["text_color"])
        self.ax.spines["right"].set_color(self.theme["text_color"])
        self.ax.spines["left"].set_color(self.theme["text_color"])

        self.ax.set_xlabel(
            "Intensidad de Pixel", color=self.theme["text_color"], fontsize=9
        )
        self.ax.set_ylabel("Frecuencia", color=self.theme["text_color"], fontsize=9)
        self.ax.grid(True, alpha=0.3, color=self.theme["text_color"])

        # Título inicial
        self.ax.set_title(
            "Carga una imagen para ver el histograma",
            color=self.theme["text_color"],
            fontsize=10,
            pad=20,
        )

        # Canvas para embedding en tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_stats_section(self):
        """Crea la sección de estadísticas"""
        stats_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        stats_frame.pack(fill="x", pady=(5, 0))

        # Título de estadísticas
        stats_title = tk.Label(
            stats_frame,
            text="📋 Estadísticas de la Imagen",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10, "bold"),
        )
        stats_title.pack(anchor="w", pady=(0, 5))

        # Frame para estadísticas en grid
        self.stats_grid = tk.Frame(stats_frame, bg=self.theme["panel_bg"])
        self.stats_grid.pack(fill="x")

        # Inicializar labels de estadísticas
        self.stats_labels = {}
        stats_items = [
            ("Media:", "mean"),
            ("Desviación:", "std"),
            ("Mínimo:", "min"),
            ("Máximo:", "max"),
            ("Mediana:", "median"),
            ("Moda:", "mode"),
        ]

        for i, (label_text, key) in enumerate(stats_items):
            # Label del nombre
            name_label = tk.Label(
                self.stats_grid,
                text=label_text,
                bg=self.theme["panel_bg"],
                fg=self.theme["text_color"],
                font=(self.fonts["main"]["family"], 8),
                width=10,
                anchor="w",
            )
            name_label.grid(
                row=i // 2, column=(i % 2) * 2, padx=(5, 2), pady=1, sticky="w"
            )

            # Label del valor
            value_label = tk.Label(
                self.stats_grid,
                text="--",
                bg=self.theme["panel_bg"],
                fg=self.theme["accent"],
                font=(self.fonts["main"]["family"], 8, "bold"),
                width=8,
                anchor="w",
            )
            value_label.grid(
                row=i // 2, column=(i % 2) * 2 + 1, padx=(2, 5), pady=1, sticky="w"
            )
            self.stats_labels[key] = value_label

        # Configurar grid
        self.stats_grid.columnconfigure(0, weight=1)
        self.stats_grid.columnconfigure(1, weight=1)
        self.stats_grid.columnconfigure(2, weight=1)
        self.stats_grid.columnconfigure(3, weight=1)

    def update_histogram(self, histogram_data: Dict[str, Any]):
        """Actualiza el histograma con nuevos datos"""
        self.current_histogram_data = histogram_data
        self.update_histogram_display()
        self.update_stats_display()

    def update_histogram_display(self):
        """Actualiza la visualización del histograma"""
        if not self.current_histogram_data:
            self.clear_histogram()
            return

        try:
            self.ax.clear()
            histogram_type = self.current_histogram_data.get("type", "grayscale")

            # Colores para los canales
            colors = {
                "red": "#FF6B6B",
                "green": "#4ECDC4",
                "blue": "#45B7D1",
                "gray": "#95A5A6",
            }

            if histogram_type == "rgb":
                # Mostrar todos los canales RGB superpuestos
                alpha = 0.7
                line_width = 1.5

                for color in ["red", "green", "blue"]:
                    if color in self.current_histogram_data:
                        data = self.current_histogram_data[color]
                        self.ax.plot(
                            data["bins"],
                            data["values"],
                            color=colors[color],
                            label=color.capitalize(),
                            linewidth=line_width,
                            alpha=alpha,
                        )

                self.ax.legend(
                    facecolor=self.theme["panel_bg"],
                    labelcolor=self.theme["text_color"],
                    fontsize=8,
                    loc="upper right",
                )

            else:  # Grayscale
                data = self.current_histogram_data.get("gray", {})
                if data:
                    # Rellenar el área bajo la curva
                    self.ax.fill_between(
                        data["bins"], data["values"], color=colors["gray"], alpha=0.6
                    )
                    # Línea principal
                    self.ax.plot(
                        data["bins"], data["values"], color=colors["gray"], linewidth=2
                    )

            # Configurar el gráfico
            self.ax.set_xlabel(
                "Intensidad de Pixel", color=self.theme["text_color"], fontsize=9
            )
            self.ax.set_ylabel("Frecuencia", color=self.theme["text_color"], fontsize=9)
            self.ax.grid(True, alpha=0.3, color=self.theme["text_color"])

            # Título dinámico
            if histogram_type == "rgb":
                title = "Histograma RGB - Canales Superpuestos"
            else:
                title = "Histograma - Escala de Grises"

            self.ax.set_title(
                title, color=self.theme["text_color"], fontsize=10, pad=10
            )

            # Ajustar límites
            self.ax.set_xlim(0, 255)

            # Actualizar canvas
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error actualizando histograma: {e}")
            self.clear_histogram()

    def update_stats_display(self):
        """Actualiza la visualización de estadísticas"""
        if (
            not self.current_histogram_data
            or "stats" not in self.current_histogram_data
        ):
            self.clear_stats()
            return

        stats = self.current_histogram_data["stats"]

        # Calcular moda si no está presente
        if "mode" not in stats:
            try:
                if "gray" in self.current_histogram_data:
                    values = self.current_histogram_data["gray"]["values"]
                    mode_index = np.argmax(values)
                    stats["mode"] = mode_index
            except:
                stats["mode"] = "--"

        # Actualizar labels
        stat_mappings = {
            "mean": stats.get("mean", "--"),
            "std": stats.get("std", "--"),
            "min": stats.get("min", "--"),
            "max": stats.get("max", "--"),
            "median": stats.get("median", "--"),
            "mode": stats.get("mode", "--"),
        }

        for key, value in stat_mappings.items():
            if isinstance(value, (int, float)):
                if key in ["mean", "std"]:
                    display_value = f"{value:.2f}"
                else:
                    display_value = f"{int(value)}"
            else:
                display_value = str(value)

            self.stats_labels[key].config(text=display_value)

    def clear_histogram(self):
        """Limpia el histograma"""
        self.ax.clear()
        self.ax.set_xlabel(
            "Intensidad de Pixel", color=self.theme["text_color"], fontsize=9
        )
        self.ax.set_ylabel("Frecuencia", color=self.theme["text_color"], fontsize=9)
        self.ax.grid(True, alpha=0.3, color=self.theme["text_color"])
        self.ax.set_title(
            "Carga una imagen para ver el histograma",
            color=self.theme["text_color"],
            fontsize=10,
            pad=20,
        )
        self.ax.set_xlim(0, 255)
        self.canvas.draw()

        self.clear_stats()

    def clear_stats(self):
        for label in self.stats_labels.values():
            label.config(text="--")

    def show_detailed_statistics(self):
        if not self.current_histogram_data:
            return

        stats_window = tk.Toplevel(self.parent)
        stats_window.title("Estadísticas Detalladas del Histograma")
        stats_window.geometry("400x500")
        stats_window.configure(bg=self.theme["panel_bg"])

        main_frame = tk.Frame(stats_window, bg=self.theme["panel_bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        title_label = tk.Label(
            main_frame,
            text="Estadísticas Detalladas",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], 14, "bold"),
        )
        title_label.pack(pady=(0, 20))

        text_widget = tk.Text(
            main_frame,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10),
            wrap="word",
            height=20,
            padx=10,
            pady=10,
        )
        text_widget.pack(fill="both", expand=True)

        stats = self.current_histogram_data.get("stats", {})
        histogram_type = self.current_histogram_data.get("type", "grayscale")

        detailed_stats = f"TIPO DE HISTOGRAMA: {histogram_type.upper()}\n"
        detailed_stats += "=" * 40 + "\n\n"

        detailed_stats += "ESTADÍSTICAS BÁSICAS:\n"
        detailed_stats += f"• Media: {stats.get('mean', 'N/A'):.2f}\n"
        detailed_stats += f"• Desviación Estándar: {stats.get('std', 'N/A'):.2f}\n"
        detailed_stats += f"• Mínimo: {stats.get('min', 'N/A')}\n"
        detailed_stats += f"• Máximo: {stats.get('max', 'N/A')}\n"
        detailed_stats += f"• Mediana: {stats.get('median', 'N/A')}\n"
        detailed_stats += f"• Moda: {stats.get('mode', 'N/A')}\n\n"

        detailed_stats += "ANÁLISIS DE DISTRIBUCIÓN:\n"
        if stats.get("mean") and stats.get("std"):
            cv = (stats["std"] / stats["mean"]) * 100 if stats["mean"] > 0 else 0
            detailed_stats += f"• Coeficiente de Variación: {cv:.2f}%\n"

            # Interpretación simple
            if stats["std"] < 20:
                detailed_stats += "• La imagen tiene bajo contraste\n"
            elif stats["std"] > 80:
                detailed_stats += "• La imagen tiene alto contraste\n"
            else:
                detailed_stats += "• La imagen tiene contraste moderado\n"

        if histogram_type == "rgb":
            detailed_stats += "\nINFORMACIÓN DE CANALES RGB:\n"
            for color in ["red", "green", "blue"]:
                if color in self.current_histogram_data:
                    channel_data = self.current_histogram_data[color]
                    channel_mean = np.mean(channel_data["values"])
                    detailed_stats += f"• {color.capitalize()}: Intensidad media ~{channel_mean:.0f}\n"

        text_widget.insert("1.0", detailed_stats)
        text_widget.config(state="disabled")

        close_btn = tk.Button(
            main_frame,
            text="Cerrar",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=stats_window.destroy,
        )
        close_btn.pack(pady=(10, 0))

    def export_histogram(self):
        if not self.current_histogram_data:
            return

        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")],
            title="Exportar histograma como...",
        )

        if file_path:
            try:
                self.figure.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=self.theme["panel_bg"],
                )
                logger.info(f"Histograma exportado: {file_path}")
            except Exception as e:
                logger.error(f"Error exportando histograma: {e}")

    def destroy(self):
        """Limpia recursos de matplotlib"""
        if self.figure:
            plt.close(self.figure)
