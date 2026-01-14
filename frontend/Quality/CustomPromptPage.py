from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout

from qfluentwidgets import Action
from qfluentwidgets import FluentIcon
from qfluentwidgets import MessageBox
from qfluentwidgets import FluentWindow
from qfluentwidgets import PlainTextEdit
from qfluentwidgets import CardWidget
from qfluentwidgets import CaptionLabel
from qfluentwidgets import StrongBodyLabel
from qfluentwidgets import CommandBar
from qfluentwidgets import SingleDirectionScrollArea

from base.Base import Base
from base.BaseLanguage import BaseLanguage
from module.Config import Config
from module.Localizer.Localizer import Localizer
from module.PromptBuilder import PromptBuilder
from widget.Separator import Separator
from widget.SwitchButtonCard import SwitchButtonCard


class CustomPromptPage(QWidget, Base):

    TEMPLATE_ITEMS = [
        ("prefix", "custom_prompt_template_prefix_title", "custom_prompt_template_prefix_desc"),
        ("base", "custom_prompt_template_base_title", "custom_prompt_template_base_desc"),
        ("suffix", "custom_prompt_template_suffix_title", "custom_prompt_template_suffix_desc"),
        ("extractor_prefix", "custom_prompt_template_extractor_prefix_title", "custom_prompt_template_extractor_prefix_desc"),
        ("extractor_base", "custom_prompt_template_extractor_base_title", "custom_prompt_template_extractor_base_desc"),
        ("extractor_suffix", "custom_prompt_template_extractor_suffix_title", "custom_prompt_template_extractor_suffix_desc"),
        ("validator", "custom_prompt_template_validator_title", "custom_prompt_template_validator_desc"),
        ("gender", "custom_prompt_template_gender_title", "custom_prompt_template_gender_desc"),
        ("translator", "custom_prompt_template_translator_title", "custom_prompt_template_translator_desc"),
        ("arbiter", "custom_prompt_template_arbiter_title", "custom_prompt_template_arbiter_desc"),
    ]

    def __init__(self, text: str, window: FluentWindow, language: BaseLanguage.Enum) -> None:
        super().__init__(window)
        self.setObjectName(text.replace(" ", "-"))
        self.fluent_window = window

        self.language = language
        if language == BaseLanguage.Enum.ZH:
            self.base_key = "custom_prompt_zh"
        else:
            self.base_key = "custom_prompt_en"

        # 载入并保存默认配置
        config = Config().load()
        self.ensure_base_template(config)
        config.save()

        # 设置主容器
        self.root = QVBoxLayout(self)
        self.root.setSpacing(8)
        self.root.setContentsMargins(24, 24, 24, 24) # 左、上、右、下

        # 创建滚动区域的内容容器
        scroll_area_vbox_widget = QWidget()
        scroll_area_vbox = QVBoxLayout(scroll_area_vbox_widget)
        scroll_area_vbox.setContentsMargins(0, 0, 0, 0)
        scroll_area_vbox.setSpacing(8)

        # 创建滚动区域
        scroll_area = SingleDirectionScrollArea(orient = Qt.Orientation.Vertical)
        scroll_area.setWidget(scroll_area_vbox_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.enableTransparentBackground()

        # 将滚动区域添加到父布局
        self.root.addWidget(scroll_area)

        # 添加控件
        self.add_widget_header(scroll_area_vbox, config, window)
        self.add_widget_body(scroll_area_vbox, config, window)

        # 填充
        scroll_area_vbox.addStretch(1)

    def ensure_base_template(self, config: Config) -> None:
        templates = self.get_templates(config)
        base_value = templates.get("base")
        legacy_value = getattr(config, f"{self.base_key}_data", None)

        if base_value is None:
            if legacy_value is not None:
                templates["base"] = legacy_value
            else:
                templates["base"] = PromptBuilder(config).get_base(self.language)

        if legacy_value is None:
            setattr(config, f"{self.base_key}_data", templates.get("base"))

    def get_templates(self, config: Config) -> dict[str, str]:
        templates = getattr(config, f"{self.base_key}_templates", None)
        if templates is None:
            templates = {}
            setattr(config, f"{self.base_key}_templates", templates)
        return templates

    def get_default_template(self, name: str) -> str:
        if name == "prefix":
            return PromptBuilder.get_prefix(self.language)
        if name == "base":
            return PromptBuilder.get_base(self.language)
        if name == "suffix":
            return PromptBuilder.get_suffix(self.language)
        if name == "extractor_prefix":
            return PromptBuilder.get_optional_template(self.language, "extractor_prefix", "prefix")
        if name == "extractor_base":
            return PromptBuilder.get_optional_template(self.language, "extractor_base", "base")
        if name == "extractor_suffix":
            return PromptBuilder.get_optional_template(self.language, "extractor_suffix", "suffix")

        try:
            return PromptBuilder.get_template(self.language, name)
        except FileNotFoundError:
            return ""

    def get_template_value(self, config: Config, name: str) -> str:
        templates = self.get_templates(config)
        if name in templates and templates.get(name) is not None:
            return templates.get(name)
        if name == "base":
            legacy_value = getattr(config, f"{self.base_key}_data", None)
            if legacy_value is not None:
                return legacy_value
        return self.get_default_template(name)

    # 头部
    def add_widget_header(self, parent: QLayout, config: Config, window: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                getattr(config, f"{self.base_key}_enable"),
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            setattr(config, f"{self.base_key}_enable", widget.get_switch_button().isChecked())
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = getattr(Localizer.get(), f"{self.base_key}_page_head"),
                description = getattr(Localizer.get(), f"{self.base_key}_page_head_desc"),
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 主体
    def add_widget_body(self, parent: QLayout, config: Config, window: FluentWindow) -> None:
        for name, title_key, desc_key in self.TEMPLATE_ITEMS:
            self.add_template_card(
                parent = parent,
                config = config,
                name = name,
                title = getattr(Localizer.get(), title_key),
                description = getattr(Localizer.get(), desc_key),
            )

    def add_template_card(self, parent: QLayout, config: Config, name: str, title: str, description: str) -> None:
        card = CardWidget(self)
        card.setBorderRadius(4)
        root = QVBoxLayout(card)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        head_container = QWidget(card)
        head_hbox = QHBoxLayout(head_container)
        head_hbox.setContentsMargins(0, 0, 0, 0)
        head_hbox.setSpacing(8)

        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(2)
        title_label = StrongBodyLabel(title, card)
        description_label = CaptionLabel(description, card)
        description_label.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        text_vbox.addWidget(title_label)
        text_vbox.addWidget(description_label)
        head_hbox.addLayout(text_vbox)
        head_hbox.addStretch(1)
        root.addWidget(head_container)

        root.addWidget(Separator(card))

        text_edit = PlainTextEdit(card)
        text_edit.setPlainText(self.get_template_value(config, name))
        text_edit.setMinimumHeight(160)
        root.addWidget(text_edit)

        command_bar = CommandBar(card)
        command_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        command_bar.addAction(Action(
            FluentIcon.SAVE,
            Localizer.get().quality_save,
            card,
            triggered = lambda: self.save_template(name, text_edit.toPlainText()),
        ))
        command_bar.addAction(Action(
            FluentIcon.CLEAR_SELECTION,
            Localizer.get().quality_reset,
            card,
            triggered = lambda: self.reset_template(name, text_edit),
        ))
        root.addWidget(command_bar)

        parent.addWidget(card)

    def save_template(self, name: str, text: str) -> None:
        config = Config().load()
        templates = self.get_templates(config)
        templates[name] = text.strip()
        setattr(config, f"{self.base_key}_templates", templates)
        if name == "base":
            setattr(config, f"{self.base_key}_data", templates[name])
        config.save()

        self.emit(Base.Event.TOAST, {
            "type": Base.ToastType.SUCCESS,
            "message": Localizer.get().quality_save_toast,
        })

    def reset_template(self, name: str, text_edit: PlainTextEdit) -> None:
        message_box = MessageBox(Localizer.get().alert, Localizer.get().quality_reset_alert, self.fluent_window)
        message_box.yesButton.setText(Localizer.get().confirm)
        message_box.cancelButton.setText(Localizer.get().cancel)

        if not message_box.exec():
            return

        default_value = self.get_default_template(name)
        text_edit.setPlainText(default_value)

        config = Config().load()
        templates = self.get_templates(config)
        templates[name] = default_value
        setattr(config, f"{self.base_key}_templates", templates)
        if name == "base":
            setattr(config, f"{self.base_key}_data", default_value)
        config.save()

        self.emit(Base.Event.TOAST, {
            "type": Base.ToastType.SUCCESS,
            "message": Localizer.get().quality_reset_toast,
        })
