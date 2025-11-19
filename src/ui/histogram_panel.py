import tkinter as tk
from typing import Any, Dict

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

        self.create_widgets()
        logger.info("HistogramPanel inicializado")

    def create_widgets(self):
        self.main_frame = tk.Frame(self.parent, bg=self.theme["panel_bg"])
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        title_label = tk.Label(
            self.main_frame,
            text="Histograma en Tiempo Real",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        )
        title_label.pack(pady=(0, 10))

        self.create_controls()

        self.create_histogram_plot()

    def create_controls(self):
        controls_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        controls_frame.pack(fill="x", pady=(0, 10))

        self.channel_var = tk.StringVar(value="all")
        channel_frame = tk.Frame(controls_frame, bg=self.theme["panel_bg"])
        channel_frame.pack(side="left", padx=5)

        tk.Label(
            channel_frame,
            text="Canal:",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
        ).pack(side="left")

        channels = [
            ("Todos", "all"),
            ("Rojo", "red"),
            ("Verde", "green"),
            ("Azul", "blue"),
            ("Gris", "gray"),
        ]
        for text, value in channels:
            tk.Radiobutton(
                channel_frame,
                text=text,
                variable=self.channel_var,
                value=value,
                bg=self.theme["panel_bg"],
                fg=self.theme["text_color"],
                selectcolor=self.theme["accent"],
                command=self.update_histogram_display,
            ).pack(side="left", padx=2)

        stats_btn = tk.Button(
            controls_frame,
            text="Estadísticas",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            command=self.show_statistics,
        )
        stats_btn.pack(side="right", padx=5)

    def create_histogram_plot(self):
        plot_frame = tk.Frame(self.main_frame, bg=self.theme["panel_bg"])
        plot_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor=self.theme["panel_bg"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.theme["panel_bg"])

        self.ax.tick_params(colors=self.theme["text_color"])
        self.ax.spines["bottom"].set_color(self.theme["text_color"])
        self.ax.spines["top"].set_color(self.theme["text_color"])
        self.ax.spines["right"].set_color(self.theme["text_color"])
        self.ax.spines["left"].set_color(self.theme["text_color"])

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.stats_text = tk.Text(
            plot_frame,
            height=4,
            width=50,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 8),
        )
        self.stats_text.pack(fill="x", pady=(5, 0))
        self.stats_text.config(state="disabled")

    def update_histogram(self, histogram_data: Dict[str, Any]):
        self.current_histogram_data = histogram_data
        self.update_histogram_display()

    def update_histogram_display(self):
        if not self.current_histogram_data:
            self.clear_histogram()
            return

        try:
            self.ax.clear()
            selected_channel = self.channel_var.get()
            histogram_type = self.current_histogram_data.get("type", "grayscale")

            colors = {
                "red": "#FF4444",
                "green": "#44FF44",
                "blue": "#4444FF",
                "gray": "#CCCCCC",
                "all": "#FFFFFF",
            }

            if histogram_type == "rgb" and selected_channel == "all":
                for color in ["red", "green", "blue"]:
                    if color in self.current_histogram_data:
                        data = self.current_histogram_data[color]
                        self.ax.plot(
                            data["bins"],
                            data["values"],
                            color=colors[color],
                            label=color.capitalize(),
                            alpha=0.7,
                        )
                self.ax.legend(
                    facecolor=self.theme["panel_bg"],
                    labelcolor=self.theme["text_color"],
                )

            elif histogram_type == "rgb" and selected_channel in [
                "red",
                "green",
                "blue",
            ]:
                data = self.current_histogram_data[selected_channel]
                self.ax.fill_between(
                    data["bins"],
                    data["values"],
                    color=colors[selected_channel],
                    alpha=0.6,
                )
                self.ax.plot(
                    data["bins"],
                    data["values"],
                    color=colors[selected_channel],
                    linewidth=1,
                )

            else:
                data = self.current_histogram_data.get("gray", {})
                if data:
                    self.ax.fill_between(
                        data["bins"], data["values"], color=colors["gray"], alpha=0.6
                    )
                    self.ax.plot(
                        data["bins"], data["values"], color=colors["gray"], linewidth=1
                    )

            self.ax.set_xlabel("Intensidad", color=self.theme["text_color"])
            self.ax.set_ylabel("Frecuencia", color=self.theme["text_color"])
            self.ax.grid(True, alpha=0.3, color=self.theme["text_color"])

            self.canvas.draw()

            self.update_stats_text()

        except Exception as e:
            logger.error(f"Error actualizando histograma: {e}")
            self.clear_histogram()

    def clear_histogram(self):
        self.ax.clear()
        self.ax.set_xlabel("Intensidad", color=self.theme["text_color"])
        self.ax.set_ylabel("Frecuencia", color=self.theme["text_color"])
        self.ax.grid(True, alpha=0.3, color=self.theme["text_color"])
        self.canvas.draw()

        # Limpiar texto de estadísticas
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "No hay datos de imagen cargada")
        self.stats_text.config(state="disabled")

    def update_stats_text(self):
        if (
            not self.current_histogram_data
            or "stats" not in self.current_histogram_data
        ):
            return

        stats = self.current_histogram_data["stats"]
        stats_text = f"""
Estadísticas de la imagen:
• Media: {stats["mean"]:.2f}
• Desviación estándar: {stats["std"]:.2f}
• Mínimo: {stats["min"]}
• Máximo: {stats["max"]}
• Mediana: {stats["median"]}
"""

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state="disabled")

    def show_statistics(self):
        if not self.current_histogram_data:
            return

        stats_window = tk.Toplevel(self.parent)
        stats_window.title("Estadísticas Detalladas")
        stats_window.geometry("300x400")
        stats_window.configure(bg=self.theme["panel_bg"])

        stats = self.current_histogram_data.get("stats", {})
        text_widget = tk.Text(
            stats_window,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["main"]["family"], 10),
        )
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        detailed_stats = "ESTADÍSTICAS DETALLADAS\n\n"
        for key, value in stats.items():
            detailed_stats += f"{key.upper()}: {value}\n"

        text_widget.insert("1.0", detailed_stats)
        text_widget.config(state="disabled")
