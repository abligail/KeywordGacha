from qfluentwidgets import Theme
from qfluentwidgets import setTheme

from module.Config import Config


def _resolve_auto_theme() -> Theme | None:
    auto_theme = getattr(Theme, "AUTO", None)
    if auto_theme is not None:
        return auto_theme
    return getattr(Theme, "SYSTEM", None)


def apply_theme(config: Config) -> None:
    theme_value = str(config.theme).upper()
    if theme_value == str(Config.Theme.DARK):
        setTheme(Theme.DARK)
        return
    if theme_value == str(Config.Theme.LIGHT):
        setTheme(Theme.LIGHT)
        return

    auto_theme = _resolve_auto_theme()
    if auto_theme is not None:
        setTheme(auto_theme)
    else:
        setTheme(Theme.LIGHT)
