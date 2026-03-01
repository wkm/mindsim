"""Driver app for MainMenuScreen snapshot test."""

from textual.app import App

from tui.screens.main_menu import MainMenuScreen


class MainMenuApp(App):
    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())


if __name__ == "__main__":
    MainMenuApp().run()
