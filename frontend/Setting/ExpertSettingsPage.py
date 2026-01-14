from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLayout
from PyQt5.QtWidgets import QVBoxLayout
from qfluentwidgets import FluentWindow
from qfluentwidgets import SingleDirectionScrollArea

from base.Base import Base
from module.Config import Config
from module.Localizer.Localizer import Localizer
from widget.SpinCard import SpinCard
from widget.SwitchButtonCard import SwitchButtonCard

class ExpertSettingsPage(QWidget, Base):

    def __init__(self, text: str, window: FluentWindow) -> None:
        super().__init__(window)
        self.setObjectName(text.replace(" ", "-"))

        # 载入并保存默认配置
        config = Config().load().save()

        # 设置容器
        self.root = QVBoxLayout(self)
        self.root.setSpacing(8)
        self.root.setContentsMargins(6, 24, 6, 24) # 左、上、右、下

        # 创建滚动区域的内容容器
        scroll_area_vbox_widget = QWidget()
        scroll_area_vbox = QVBoxLayout(scroll_area_vbox_widget)
        scroll_area_vbox.setContentsMargins(18, 0, 18, 0)

        # 创建滚动区域
        scroll_area = SingleDirectionScrollArea(orient = Qt.Orientation.Vertical)
        scroll_area.setWidget(scroll_area_vbox_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.enableTransparentBackground()

        # 将滚动区域添加到父布局
        self.root.addWidget(scroll_area)

        # 添加控件
        self.add_widget_multi_agent_enable(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_review_output(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_context_window(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_context_budget(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_context_budget_long(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_gender_retry_long(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_gender_vote_min_count(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_gender_vote_max_windows(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_gender_high_confidence_min_count(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_review_high_freq_min_count(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_title_filter(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_review_arbitrate(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_review_arbitrate_apply(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_translate_post(scroll_area_vbox, config, window)
        self.add_widget_multi_agent_apply_on_export(scroll_area_vbox, config, window)
        self.add_widget_output_choices(scroll_area_vbox, config, window)
        self.add_widget_output_kvjson(scroll_area_vbox, config, window)

        # 填充
        scroll_area_vbox.addStretch(1)

    # 输出候选数据
    def add_widget_output_choices(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.output_choices
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.output_choices = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_output_choices_title,
                description = Localizer.get().expert_settings_page_output_choices_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 多 Agent 流程
    def add_widget_multi_agent_enable(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_enable
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_enable = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_title,
                description = Localizer.get().expert_settings_page_multi_agent_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 输出复核文件
    def add_widget_multi_agent_review_output(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_review_output
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_review_output = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_review_title,
                description = Localizer.get().expert_settings_page_multi_agent_review_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # snippet 窗口行数
    def add_widget_multi_agent_context_window(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 20)
            widget.get_spin_box().setValue(config.multi_agent_context_window)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_context_window = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_context_window_title,
                description = Localizer.get().expert_settings_page_multi_agent_context_window_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # snippet 短上下文预算
    def add_widget_multi_agent_context_budget(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 9999999)
            widget.get_spin_box().setValue(config.multi_agent_context_budget)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_context_budget = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_context_budget_title,
                description = Localizer.get().expert_settings_page_multi_agent_context_budget_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # snippet 长上下文预算
    def add_widget_multi_agent_context_budget_long(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 9999999)
            widget.get_spin_box().setValue(config.multi_agent_context_budget_long)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_context_budget_long = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_context_budget_long_title,
                description = Localizer.get().expert_settings_page_multi_agent_context_budget_long_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # 性别判定低置信度重试
    def add_widget_multi_agent_gender_retry_long(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_gender_retry_long
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_gender_retry_long = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_gender_retry_title,
                description = Localizer.get().expert_settings_page_multi_agent_gender_retry_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 性别判定多窗口触发阈值
    def add_widget_multi_agent_gender_vote_min_count(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 9999999)
            widget.get_spin_box().setValue(config.multi_agent_gender_vote_min_count)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_gender_vote_min_count = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_gender_vote_min_count_title,
                description = Localizer.get().expert_settings_page_multi_agent_gender_vote_min_count_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # 性别判定多窗口最大数量
    def add_widget_multi_agent_gender_vote_max_windows(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 99)
            widget.get_spin_box().setValue(config.multi_agent_gender_vote_max_windows)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_gender_vote_max_windows = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_gender_vote_max_windows_title,
                description = Localizer.get().expert_settings_page_multi_agent_gender_vote_max_windows_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # 性别判定高置信度计数阈值
    def add_widget_multi_agent_gender_high_confidence_min_count(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 9999999)
            widget.get_spin_box().setValue(config.multi_agent_gender_high_confidence_min_count)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_gender_high_confidence_min_count = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_gender_high_confidence_min_count_title,
                description = Localizer.get().expert_settings_page_multi_agent_gender_high_confidence_min_count_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # 高频证据不足复核阈值
    def add_widget_multi_agent_review_high_freq_min_count(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SpinCard) -> None:
            widget.get_spin_box().setRange(0, 9999999)
            widget.get_spin_box().setValue(config.multi_agent_review_high_freq_min_count)

        def value_changed(widget: SpinCard) -> None:
            config = Config().load()
            config.multi_agent_review_high_freq_min_count = widget.get_spin_box().value()
            config.save()

        parent.addWidget(
            SpinCard(
                title = Localizer.get().expert_settings_page_multi_agent_review_high_freq_min_count_title,
                description = Localizer.get().expert_settings_page_multi_agent_review_high_freq_min_count_description,
                init = init,
                value_changed = value_changed,
            )
        )

    # 称谓/头衔硬过滤
    def add_widget_multi_agent_title_filter(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_title_filter_enable
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_title_filter_enable = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_title_filter_title,
                description = Localizer.get().expert_settings_page_multi_agent_title_filter_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 复核仲裁
    def add_widget_multi_agent_review_arbitrate(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_review_arbitrate
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_review_arbitrate = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_review_arbitrate_title,
                description = Localizer.get().expert_settings_page_multi_agent_review_arbitrate_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 复核仲裁回填
    def add_widget_multi_agent_review_arbitrate_apply(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_review_arbitrate_apply
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_review_arbitrate_apply = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_review_arbitrate_apply_title,
                description = Localizer.get().expert_settings_page_multi_agent_review_arbitrate_apply_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 翻译后置
    def add_widget_multi_agent_translate_post(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_translate_post
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_translate_post = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_translate_post_title,
                description = Localizer.get().expert_settings_page_multi_agent_translate_post_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 导出时执行多 Agent
    def add_widget_multi_agent_apply_on_export(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.multi_agent_apply_on_export
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.multi_agent_apply_on_export = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_multi_agent_export_title,
                description = Localizer.get().expert_settings_page_multi_agent_export_description,
                init = init,
                checked_changed = checked_changed,
            )
        )

    # 输出 KVJSON 文件
    def add_widget_output_kvjson(self, parent: QLayout, config: Config, windows: FluentWindow) -> None:

        def init(widget: SwitchButtonCard) -> None:
            widget.get_switch_button().setChecked(
                config.output_kvjson
            )

        def checked_changed(widget: SwitchButtonCard) -> None:
            config = Config().load()
            config.output_kvjson = widget.get_switch_button().isChecked()
            config.save()

        parent.addWidget(
            SwitchButtonCard(
                title = Localizer.get().expert_settings_page_output_kvjson_title,
                description = Localizer.get().expert_settings_page_output_kvjson_description,
                init = init,
                checked_changed = checked_changed,
            )
        )
