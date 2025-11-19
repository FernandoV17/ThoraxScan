import tkinter as tk

from src.ui.controller import UIStyleManager
from src.ui.loading_screen import LoadingScreen
from src.ui.main_window import MainWindow


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.styles = UIStyleManager()

        self.show_loading_screen()

    def show_loading_screen(self):
        self.splash = LoadingScreen(self.root)

        self.root.after(1500, self.start_main_window)

    def start_main_window(self):
        self.splash.destroy()

        self.root.deiconify()
        MainWindow(self.root)

    def run(self):
        self.root.mainloop()
