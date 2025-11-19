import tkinter as tk

from src.ui.controllers.version_controller import VersionController


class WelcomePopup:
    def __init__(self, root, theme, fonts, version_object, on_close=None):
        # Asignaciones correctas
        self.root = root
        self.theme = theme
        self.fonts = fonts
        self.version = version_object
        self.on_close = on_close

        msg = f"""
            Gracias por usar {self.version.get_name()}.
            Esta es la versión {self.version.get_full_version()}.

            Versión: {self.version.get_version()}
            Release: {self.version.get_release()}
            Autor: {self.version.get_author().get("main")}

            Agradecemos su preferencia y apoyo para seguir mejorando. Esperen
            novedades pronto, por el momento pedimos paciencia y que si
            tienen ideas, sugerencias, comentarios o errores pueden reportarlos
            via GitHub.
            Gracias por usar nuestro programa.
            -- Atte : {self.version.get_author().get("main")}
        """.strip()

        # Ahora sí puedes usar self.root
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Bienvenido")
        self.popup.transient(self.root)
        self.popup.grab_set()
        self.popup.geometry("420x350")
        self.popup.configure(bg=self.theme["panel_bg"])

        tk.Label(
            self.popup,
            text="Gracias por usar el programa",
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            font=(self.fonts["title"]["family"], self.fonts["title"]["size"], "bold"),
        ).pack(pady=(10, 6))

        tk.Label(
            self.popup,
            text=msg,
            bg=self.theme["panel_bg"],
            fg=self.theme["text_color"],
            justify="left",
        ).pack(padx=15, pady=5)

        tk.Button(
            self.popup,
            text="OK",
            bg=self.theme["accent"],
            fg=self.theme["text_color"],
            bd=0,
            command=self.close,
        ).pack(pady=10)

    def close(self):
        if self.on_close:
            self.on_close()
        self.popup.grab_release()
        self.popup.destroy()
