import concurrent.futures
import copy
import os
import re
import shutil
import threading
import time
import webbrowser

import httpx
import opencc
from rich.progress import TaskID

from base.Base import Base
from base.BaseLanguage import BaseLanguage
from base.LogManager import LogManager
from model.Item import Item
from module.CacheManager import CacheManager
from module.Config import Config
from module.Engine.Engine import Engine
from module.Engine.NERAnalyzer.NERAnalyzerTask import NERAnalyzerTask
from module.Engine.TaskLimiter import TaskLimiter
from module.Engine.TaskRequester import TaskRequester
from module.FakeNameHelper import FakeNameHelper
from module.File.FileManager import FileManager
from module.Filter.LanguageFilter import LanguageFilter
from module.Filter.RuleFilter import RuleFilter
from module.Filter.TitleFilter import TitleFilter
from module.Localizer.Localizer import Localizer
from module.Normalizer import Normalizer
from module.ProgressBar import ProgressBar
from module.PromptBuilder import PromptBuilder
from module.Response.ResponseDecoder import ResponseDecoder
from module.RubyCleaner import RubyCleaner
from module.Text.TextHelper import TextHelper

class NERAnalyzer(Base):

    BLACKLIST_INFO: set[str] = {
        "其它",
        "其他",
        "other",
        "others",
    }

    TARGET_MARKER_START: str = "【TARGET】"
    TARGET_MARKER_END: str = "【/TARGET】"
    GENDER_CLUES: tuple[tuple[str, int]] = (
        ("彼女", 3),
        ("彼", 2),
        ("妻", 3),
        ("夫", 3),
        ("母", 3),
        ("父", 3),
        ("姉", 3),
        ("兄", 3),
        ("少女", 3),
        ("少年", 3),
        ("女の子", 3),
        ("男の子", 3),
        ("王女", 3),
        ("王子", 3),
        ("女", 2),
        ("男", 2),
        ("俺", 1),
        ("僕", 1),
        ("私", 1),
        ("あたし", 1),
        ("くん", 1),
        ("ちゃん", 1),
    )
    VALIDATOR_RETRY_KEYWORDS: tuple[str, ...] = (
        "证据不足",
        "上下文不足",
        "证据不够",
        "信息不足",
        "证据冲突",
        "insufficient",
        "not enough",
        "ambiguous",
        "uncertain",
        "conflict",
        "low confidence",
    )

    # 类变量
    OPENCCT2S: opencc.OpenCC = opencc.OpenCC("t2s")
    OPENCCS2T: opencc.OpenCC = opencc.OpenCC("s2tw")

    def __init__(self) -> None:
        super().__init__()

        # 初始化
        self.cache_manager = CacheManager(service = True)

        # 线程锁
        self.lock = threading.Lock()
        self.agent_usage_lock = threading.Lock()
        self.agent_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # 注册事件
        self.subscribe(Base.Event.PROJECT_CHECK_RUN, self.project_check_run)
        self.subscribe(Base.Event.NER_ANALYZER_RUN, self.ner_analyzer_run)
        self.subscribe(Base.Event.NER_ANALYZER_EXPORT, self.ner_analyzer_export)
        self.subscribe(Base.Event.NER_ANALYZER_REQUIRE_STOP, self.ner_analyzer_require_stop)

    # 项目检查事件
    def project_check_run(self, event: Base.Event, data: dict) -> None:

        def task(event: str, data: dict) -> None:
            if Engine.get().get_status() != Base.TaskStatus.IDLE:
                status = Base.ProjectStatus.NONE
            else:
                cache_manager = CacheManager(service = False)
                cache_manager.load_project_from_file(Config().load().output_folder)
                status = cache_manager.get_project().get_status()

            self.emit(Base.Event.PROJECT_CHECK_DONE, {
                "status" : status,
            })
        threading.Thread(target = task, args = (event, data)).start()

    # 运行事件
    def ner_analyzer_run(self, event: Base.Event, data: dict) -> None:
        if Engine.get().get_status() == Base.TaskStatus.IDLE:
            threading.Thread(
                target = self.start,
                args = (event, data),
            ).start()
        else:
            self.emit(Base.Event.TOAST, {
                "type": Base.ToastType.WARNING,
                "message": Localizer.get().engine_task_running,
            })

    # 停止事件
    def ner_analyzer_export(self, event: Base.Event, data: dict) -> None:
        if Engine.get().get_status() != Base.TaskStatus.NERING:
            return None

        # 复制一份以避免影响原始数据
        def task(event: str, data: dict) -> None:
            self.save_ouput(
                copy.deepcopy(self.cache_manager.get_project().get_extras().get("glossary", [])),
                end = False,
            )
        threading.Thread(target = task, args = (event, data)).start()

    # 请求停止事件
    def ner_analyzer_require_stop(self, event: Base.Event, data: dict) -> None:
        Engine.get().set_status(Base.TaskStatus.STOPPING)

        def task(event: str, data: dict) -> None:
            while True:
                time.sleep(0.5)

                if Engine.get().get_running_task_count() == 0:
                    # 等待回调执行完毕
                    time.sleep(1.0)

                    # 写入缓存
                    self.cache_manager.save_to_file(
                        project = self.cache_manager.get_project(),
                        items = self.cache_manager.get_items(),
                        output_folder = self.config.output_folder,
                    )

                    # 日志
                    self.print("")
                    self.info(Localizer.get().engine_task_stop)
                    self.print("")

                    # 通知
                    self.emit(Base.Event.TOAST, {
                        "type": Base.ToastType.SUCCESS,
                        "message": Localizer.get().engine_task_stop,
                    })

                    # 更新运行状态
                    Engine.get().set_status(Base.TaskStatus.IDLE)
                    self.emit(Base.Event.NER_ANALYZER_DONE, {})
                    break
        threading.Thread(target = task, args = (event, data)).start()

    # 开始
    def start(self, event: Base.Event, data: dict) -> None:
        config: Base.ProjectStatus = data.get("config")
        status: Base.ProjectStatus = data.get("status")

        # 更新运行状态
        Engine.get().set_status(Base.TaskStatus.NERING)

        # 初始化
        self.config = config if isinstance(config, Config) else Config().load()
        self.platform = self.config.get_platform(self.config.activate_platform)
        max_workers, rpm_threshold = self.initialize_max_workers()

        # 重置
        TaskRequester.reset()
        PromptBuilder.reset()
        FakeNameHelper.reset()

        # 生成缓存列表
        if status == Base.ProjectStatus.PROCESSING:
            self.cache_manager.load_from_file(self.config.output_folder)
        else:
            shutil.rmtree(f"{self.config.output_folder}/cache", ignore_errors = True)
            project, items = FileManager(self.config).read_from_path()
            self.cache_manager.set_items(items)
            self.cache_manager.set_project(project)

        # 检查数据是否为空
        if self.cache_manager.get_item_count() == 0:
            # 通知
            self.emit(Base.Event.TOAST, {
                "type": Base.ToastType.WARNING,
                "message": Localizer.get().engine_no_items,
            })

            self.emit(Base.Event.NER_ANALYZER_REQUIRE_STOP, {})
            return None

        # 兼容性处理
        for item in self.cache_manager.get_items():
            if item.get_status() == Base.ProjectStatus.PROCESSED_IN_PAST:
                item.set_status(Base.ProjectStatus.NONE)

        # 从头翻译时加载默认数据
        if status == Base.ProjectStatus.PROCESSING:
            self.extras = self.cache_manager.get_project().get_extras()
            self.extras["start_time"] = time.time() - self.extras.get("time", 0)
        else:
            self.extras = {
                "start_time": time.time(),
                "total_line": 0,
                "line": 0,
                "total_tokens": 0,
                "total_output_tokens": 0,
                "time": 0,
                "glossary": [],
                "stage": "extracting",
                "stage_progress": 0,
                "stage_total": 0,
            }

        # 更新翻译进度
        self.emit(Base.Event.NER_ANALYZER_UPDATE, self.extras)

        # 规则过滤
        self.rule_filter(self.cache_manager.get_items())

        # 语言过滤
        self.language_filter(self.cache_manager.get_items())

        # 开始循环
        for current_round in range(self.config.max_round):
            # 检测是否需要停止任务
            # 目的是避免用户正好在两轮之间停止任务
            if Engine.get().get_status() == Base.TaskStatus.STOPPING:
                return None

            # 第一轮且不是继续翻译时，记录任务的总行数
            if current_round == 0 and status == Base.ProjectStatus.NONE:
                self.extras["total_line"] = self.cache_manager.get_item_count_by_status(Base.ProjectStatus.NONE)

            # 第二轮开始切分
            if current_round > 0:
                self.config.token_threshold = max(1, int(self.config.token_threshold / 2))

            # 生成缓存数据条目片段
            chunks = self.cache_manager.generate_item_chunks(self.config.token_threshold)

            # 生成翻译任务
            self.print("")
            tasks: list[NERAnalyzerTask] = []
            with ProgressBar(transient = False) as progress:
                pid = progress.new()
                for items in chunks:
                    progress.update(pid, advance = 1, total = len(chunks))
                    tasks.append(NERAnalyzerTask(self.config, self.platform, items))

            # 打印日志
            self.info(Localizer.get().engine_task_generation.replace("{COUNT}", str(len(chunks))))

            # 输出开始翻译的日志
            self.print("")
            self.print("")
            self.info(f"{Localizer.get().engine_current_round} - {current_round + 1}")
            self.info(f"{Localizer.get().engine_max_round} - {self.config.max_round}")
            self.print("")
            self.info(f"{Localizer.get().engine_api_name} - {self.platform.get('name')}")
            self.info(f"{Localizer.get().engine_api_url} - {self.platform.get('api_url')}")
            self.info(f"{Localizer.get().engine_api_model} - {self.platform.get('model')}")
            self.print("")
            task_type = "extractor" if self.config.multi_agent_enable == True and self.config.multi_agent_translate_post == True else None
            self.info(PromptBuilder(self.config).build_main(task_type))
            self.print("")

            # 开始执行翻译任务
            task_limiter = TaskLimiter(rps = max_workers, rpm = rpm_threshold)
            with ProgressBar(transient = True) as progress:
                with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers, thread_name_prefix = Engine.TASK_PREFIX) as executor:
                    pid = progress.new()
                    for task in tasks:
                        # 检测是否需要停止任务
                        # 目的是绕过限流器，快速结束所有剩余任务
                        if Engine.get().get_status() == Base.TaskStatus.STOPPING:
                            return None

                        task_limiter.wait()
                        future = executor.submit(task.start)
                        future.add_done_callback(lambda future: self.task_done_callback(future, pid, progress))

            # 判断是否需要继续翻译
            if self.cache_manager.get_item_count_by_status(Base.ProjectStatus.NONE) == 0:
                self.cache_manager.get_project().set_status(Base.ProjectStatus.PROCESSED)

                # 日志
                self.print("")
                self.info(Localizer.get().engine_task_done)
                self.info(Localizer.get().engine_task_save)

                # 通知
                self.emit(Base.Event.TOAST, {
                    "type": Base.ToastType.SUCCESS,
                    "message": Localizer.get().engine_task_done,
                })
                break

            # 检查是否达到最大轮次
            if current_round >= self.config.max_round - 1:
                # 日志
                self.print("")
                self.warning(Localizer.get().engine_task_fail)
                self.warning(Localizer.get().engine_task_save)

                # 通知
                self.emit(Base.Event.TOAST, {
                    "type": Base.ToastType.SUCCESS,
                    "message": Localizer.get().engine_task_fail,
                })
                break

        # 等待回调执行完毕
        time.sleep(1.0)

        # 写入缓存
        self.cache_manager.save_to_file(
            project = self.cache_manager.get_project(),
            items = self.cache_manager.get_items(),
            output_folder = self.config.output_folder,
        )

        # 检查结果并写入文件
        self.save_ouput(
            self.cache_manager.get_project().get_extras().get("glossary", []),
            end = True,
        )

        # 重置内部状态（正常完成翻译）
        Engine.get().set_status(Base.TaskStatus.IDLE)

        # 触发翻译停止完成的事件
        self.emit(Base.Event.NER_ANALYZER_DONE, {})

    # 初始化速度控制器
    def initialize_max_workers(self) -> tuple[int, int]:
        max_workers: int = self.config.max_workers
        rpm_threshold: int = self.config.rpm_threshold

        # 当 max_workers = 0 时，尝试获取 llama.cpp 槽数
        if max_workers == 0:
            try:
                response_json = None
                response = httpx.get(re.sub(r"/v1$", "", self.platform.get("api_url")) + "/slots")
                response.raise_for_status()
                response_json = response.json()
            except Exception:
                pass
            if isinstance(response_json, list) and len(response_json) > 0:
                max_workers = len(response_json)

        if max_workers == 0 and rpm_threshold == 0:
            max_workers = 8
            rpm_threshold = 0
        elif max_workers > 0 and rpm_threshold == 0:
            pass
        elif max_workers == 0 and rpm_threshold > 0:
            max_workers = 8192
            rpm_threshold = rpm_threshold

        return max_workers, rpm_threshold

    # 规则过滤
    def rule_filter(self, items: list[Item]) -> None:
        if len(items) == 0:
            return None

        # 筛选
        self.print("")
        count: int = 0
        with ProgressBar(transient = False) as progress:
            pid = progress.new()
            for item in items:
                progress.update(pid, advance = 1, total = len(items))
                if RuleFilter.filter(item.get_src()) == True:
                    count = count + 1
                    item.set_status(Base.ProjectStatus.EXCLUDED)

        # 打印日志
        self.info(Localizer.get().engine_task_rule_filter.replace("{COUNT}", str(count)))

    # 语言过滤
    def language_filter(self, items: list[Item]) -> None:
        if len(items) == 0:
            return None

        # 筛选
        self.print("")
        count: int = 0
        with ProgressBar(transient = False) as progress:
            pid = progress.new()
            for item in items:
                progress.update(pid, advance = 1, total = len(items))
                if LanguageFilter.filter(item.get_src(), self.config.source_language) == True:
                    count = count + 1
                    item.set_status(Base.ProjectStatus.EXCLUDED)

        # 打印日志
        self.info(Localizer.get().engine_task_language_filter.replace("{COUNT}", str(count)))

    # 输出结果
    def save_ouput(self, glossary: list[dict[str, str]], end: bool) -> None:
        apply_multi_agent = self.config.multi_agent_enable == True and (
            end == True
            or self.config.multi_agent_apply_on_export == True
            or self.config.multi_agent_translate_post == True
        )
        allow_empty_dst = apply_multi_agent == True and self.config.multi_agent_translate_post == True

        group: dict[str, list[dict[str, str]]] = {}
        with self.lock:
            v: dict[str, str] = {}
            for v in glossary:
                src: str = v.get("src").strip()
                dst: str = v.get("dst").strip()
                info: str = v.get("info").strip()

                # 简繁转换
                dst = self.convert_chinese_character_form(dst)
                info = self.convert_chinese_character_form(info)

                # 伪名还原
                src, fake_name_injected = FakeNameHelper.restore(src)

                # 将原文和译文都按标点切分
                srcs: list[str] = TextHelper.split_by_punctuation(src, split_by_space = True)
                dsts: list[str] = TextHelper.split_by_punctuation(dst, split_by_space = True)
                if len(srcs) != len(dsts):
                    srcs = [src]
                    dsts = [dst]
                for src, dst in zip(srcs, dsts):
                    src = src.strip()
                    dst = dst.strip()

                    if fake_name_injected == True:
                        continue
                    elif src == "" or (dst == "" and allow_empty_dst == False):
                        continue
                    elif src == dst and info == "":
                        continue
                    elif self.check(src, dst, info) == False:
                        continue

                    group.setdefault(src, []).append({
                        "src": src,
                        "dst": dst,
                        "info": info,
                    })

        glossary: list[dict[str, str]] = []
        for src, choices in group.items():
            glossary.append(self.find_best(src, choices))

        # 去重
        glossary = list({v.get("src"): v for v in glossary}.values())

        # 计数
        glossary = self.search_for_context(glossary, self.cache_manager.get_items(), end, build_snippets = apply_multi_agent)

        review_entries: list[dict[str, str]] = []
        if apply_multi_agent == True:
            glossary, review_entries = self.apply_multi_agent_pipeline(glossary)

        # 排序
        def count_sort_key(entry: dict[str, str | int | list[str]]) -> tuple[int, str]:
            count = entry.get("count", 0)
            try:
                count_value = int(count)
            except Exception:
                count_value = 0
            return (-count_value, str(entry.get("src", "")))

        glossary = sorted(glossary, key = count_sort_key)

        # 写入文件
        file_manager = FileManager(self.config)
        file_manager.write_to_path(glossary)
        if (
            self.config.multi_agent_enable == True
            and end == True
            and self.config.multi_agent_review_output == True
            and review_entries != []
        ):
            review_entries = sorted(review_entries, key = lambda x: x.get("src", ""))
            file_manager.write_review_to_path(review_entries)
        self.print("")
        self.info(Localizer.get().engine_task_save_done.replace("{PATH}", self.config.output_folder))
        self.print("")

        # 打开输出文件夹
        if self.config.output_folder_open_on_finish == True:
            webbrowser.open(os.path.abspath(self.config.output_folder))

    # 有效性检查
    def check(self, src: str, dst: str, info: str) -> bool:
        result: bool = True

        if TextHelper.get_display_lenght(src) > 32:
            result = False
        elif info.lower() in __class__.BLACKLIST_INFO:
            result = False

        return result

    # 找出最佳结果
    def find_best(self, src: str, choices: list[dict[str, str]]) -> dict[str, str]:
        dst_count: dict[str, int] = {}
        dst_choices: set[str] = set()
        for choice in choices:
            dst: str = choice.get("dst")
            dst_choices.add(dst)
            dst_count[dst] = dst_count.setdefault(dst, 0) + 1
        dst = max(dst_count, key = dst_count.get)

        info_count: dict[str, int] = {}
        info_choices: set[str] = set()
        for choice in choices:
            info: str = choice.get("info")
            info_choices.add(info)
            info_count[info] = info_count.setdefault(info, 0) + 1
        info = max(info_count, key = info_count.get)

        return {
            "src": src,
            "dst": dst,
            "dst_choices": dst_choices,
            "info": info,
            "info_choices": info_choices,
        }

    def apply_pre_replacement(self, src: str) -> str:
        if self.config.pre_replacement_enable == False:
            return src

        for v in self.config.pre_replacement_data:
            if v.get("regex", False) != True:
                src = src.replace(v.get("src"), v.get("dst"))
            else:
                src = re.sub(rf"{v.get('src')}", rf"{v.get('dst')}", src)

        return src

    def normalize_context_text(self, src: str) -> str:
        if not isinstance(src, str):
            return ""

        text = Normalizer.normalize(src)
        text = RubyCleaner.clean(text)
        text = self.apply_pre_replacement(text)

        return text

    # 搜索参考文本，并按出现次数排序
    def search_for_context(
        self,
        glossary: list[dict[str, str]],
        items: list[Item],
        end: bool,
        build_snippets: bool | None = None,
    ) -> list[dict[str, str | int | list[str]]]:
        if build_snippets is None:
            build_snippets = self.config.multi_agent_enable == True and end == True

        lines_info: list[tuple[str, str]] = [
            (item.get_file_path(), item.get_src().strip())
            for item in items
            if item.get_status() == Base.ProjectStatus.PROCESSED
        ]
        raw_lines: list[str] = [line for _, line in lines_info]
        raw_lines_cp: list[str] = raw_lines.copy()
        normalized_lines: list[str] = [self.normalize_context_text(line) for line in raw_lines]
        normalized_lines_cp: list[str] = normalized_lines.copy()
        match_lines_raw: list[str] = raw_lines.copy()
        match_lines_norm: list[str] = normalized_lines.copy()
        file_positions: dict[str, list[int]] = {}
        file_index_map: dict[int, int] = {}
        for idx, (file_path, _) in enumerate(lines_info):
            positions = file_positions.setdefault(file_path, [])
            positions.append(idx)
            file_index_map[idx] = len(positions) - 1

        # 按实体词语的长度降序排序
        glossary = sorted(glossary, key = lambda x: len(x.get("src")), reverse = True)

        self.print("")
        with ProgressBar(transient = False) as progress:
            pid = progress.new() if end == True else None
            for entry in glossary:
                progress.update(pid, advance = 1, total = len(glossary)) if end == True else None
                src: str = entry.get("src")
                raw_src = src.strip()
                normalized_src = self.normalize_context_text(raw_src)

                # 找出匹配的行
                index: set[int] = set()
                use_normalized = False
                if normalized_src != "":
                    index = {i for i, line in enumerate(match_lines_norm) if normalized_src in line}
                    use_normalized = index != set()
                if index == set() and raw_src != "":
                    index = {i for i, line in enumerate(match_lines_raw) if raw_src in line}
                    use_normalized = False

                # 获取匹配的参考文本，去重，并按长度降序排序
                entry["context"] = sorted(
                    list({line for i, line in enumerate(raw_lines_cp) if i in index}),
                    key = lambda x: len(x),
                    reverse = True,
                )
                entry["count"] = len(entry.get("context"))

                if build_snippets == True:
                    snippet_lines = normalized_lines_cp if use_normalized == True else raw_lines_cp
                    target_src = normalized_src if use_normalized == True else raw_src
                    entry["snippets"] = self.build_snippets(
                        target_src,
                        index,
                        lines_info,
                        snippet_lines,
                        file_positions,
                        file_index_map,
                    )

                # 掩盖已命中的实体词语文本，避免其子串错误的与父串匹配
                if index != set():
                    if normalized_src != "":
                        match_lines_norm = [
                            line.replace(normalized_src, len(normalized_src) * "#") if i in index else line
                            for i, line in enumerate(match_lines_norm)
                        ]
                    if raw_src != "":
                        match_lines_raw = [
                            line.replace(raw_src, len(raw_src) * "#") if i in index else line
                            for i, line in enumerate(match_lines_raw)
                        ]

        # 打印日志
        self.info(Localizer.get().engine_task_context_search.replace("{COUNT}", str(len(glossary))))

        # 按出现次数降序排序
        return sorted(glossary, key = lambda x: x.get("count"), reverse = True)

    def build_snippets(
        self,
        src: str,
        indices: set[int],
        lines_info: list[tuple[str, str]],
        lines_text: list[str],
        file_positions: dict[str, list[int]],
        file_index_map: dict[int, int],
    ) -> list[dict[str, str | int]]:
        if not isinstance(src, str) or src == "":
            return []

        window = max(0, int(self.config.multi_agent_context_window))
        snippets: list[dict[str, str | int]] = []
        seen: set[str] = set()

        for idx in sorted(indices):
            file_path = lines_info[idx][0]
            positions = file_positions.get(file_path, [])
            local_pos = file_index_map.get(idx, 0)
            start = max(0, local_pos - window)
            end = min(len(positions) - 1, local_pos + window)
            snippet_indices = positions[start : end + 1] if positions != [] else [idx]

            snippet_lines = [lines_text[i] for i in snippet_indices]
            snippet_text = "\n".join(snippet_lines)
            highlighted = snippet_text.replace(
                src,
                f"{__class__.TARGET_MARKER_START}{src}{__class__.TARGET_MARKER_END}",
            )
            if highlighted in seen:
                continue

            seen.add(highlighted)
            snippets.append(
                {
                    "text": highlighted,
                    "score": self.score_snippet(highlighted),
                    "order": snippet_indices[0] if snippet_indices != [] else idx,
                    "file": file_path,
                }
            )

        return snippets

    def score_snippet(self, text: str) -> int:
        score = 0
        for token, weight in __class__.GENDER_CLUES:
            if token in text:
                score += weight * text.count(token)
        return score

    def format_snippet_context(self, snippets: list[dict[str, str | int]], budget: int, prompt_language: BaseLanguage.Enum) -> str:
        if snippets == []:
            return "（无上下文）" if prompt_language == BaseLanguage.Enum.ZH else "No context."

        budget = max(0, int(budget))
        long_budget = max(0, int(self.config.multi_agent_context_budget_long))
        is_long = long_budget > 0 and budget >= long_budget
        window = max(0, int(self.config.multi_agent_context_window))
        min_distance = max(1, window * 2 + 1)
        similarity_threshold = 0.88
        if is_long == True:
            min_distance = max(1, window)
            similarity_threshold = 0.95

        sorted_snippets = sorted(
            snippets,
            key = lambda x: (-int(x.get("score", 0)), int(x.get("order", 0))),
        )

        blocks: list[str] = []
        selected: list[dict[str, str | int]] = []
        used = 0
        for snippet in sorted_snippets:
            text = str(snippet.get("text", ""))
            if text == "":
                continue
            if self.is_snippet_too_similar(text, selected, similarity_threshold):
                continue
            if self.is_snippet_too_close(snippet, selected, min_distance):
                continue

            block = f"[S{len(blocks) + 1:03d}]\n{text}"
            if blocks != [] and used + len(block) > budget:
                break
            blocks.append(block)
            selected.append(snippet)
            used += len(block)

        if blocks == [] and sorted_snippets != []:
            text = str(sorted_snippets[0].get("text", ""))
            blocks.append(f"[S001]\n{text}")

        return "\n\n".join(blocks)

    def get_gender_vote_window_count(self, entry: dict[str, str | int | list[str]]) -> int:
        threshold = max(0, int(self.config.multi_agent_gender_vote_min_count))
        max_windows = max(0, int(self.config.multi_agent_gender_vote_max_windows))
        if threshold <= 0 or max_windows <= 1:
            return 0
        count = entry.get("count", 0)
        try:
            count_value = int(count)
        except Exception:
            count_value = 0
        if count_value < threshold:
            return 0
        snippets = entry.get("snippets", [])
        if not isinstance(snippets, list) or len(snippets) < 2:
            return 0
        windows = max(2, (count_value + threshold - 1) // threshold)
        windows = min(max_windows, windows, len(snippets))
        return windows

    def split_snippets_for_gender_vote(
        self,
        snippets: list[dict[str, str | int]],
        windows: int,
    ) -> list[list[dict[str, str | int]]]:
        if snippets == []:
            return []
        if windows <= 1:
            return [snippets]
        ordered = sorted(snippets, key = lambda x: int(x.get("order", 0)))
        chunk_size = max(1, (len(ordered) + windows - 1) // windows)
        groups = [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]
        return groups if groups != [] else [snippets]

    def count_evidence_refs(self, evidence: str) -> int:
        if not isinstance(evidence, str) or evidence == "":
            return 0
        return len(re.findall(r"S\d{3}", evidence, flags = re.IGNORECASE))

    def score_gender_result(self, result: dict[str, str]) -> int:
        score = 0
        if result.get("confidence", "") == "high":
            score += 10
        if self.is_evidence_valid(result.get("evidence", "")):
            score += 5
        score += self.count_evidence_refs(result.get("evidence", ""))
        return score

    def build_gender_vote_summary(self, results: list[dict[str, str]], labels: dict[str, str]) -> str:
        if results == []:
            return ""
        counts = {
            "male": {"high": 0, "low": 0},
            "female": {"high": 0, "low": 0},
            "unknown": {"high": 0, "low": 0},
            "invalid": 0,
        }
        details: list[str] = []
        for idx, result in enumerate(results, start = 1):
            gender = result.get("gender", "")
            confidence = result.get("confidence", "")
            evidence = result.get("evidence", "")
            if gender == labels.get("male"):
                counts["male"]["high" if confidence == "high" else "low"] += 1
                gender_label = "male"
            elif gender == labels.get("female"):
                counts["female"]["high" if confidence == "high" else "low"] += 1
                gender_label = "female"
            elif gender == labels.get("unknown"):
                counts["unknown"]["high" if confidence == "high" else "low"] += 1
                gender_label = "unknown"
            else:
                counts["invalid"] += 1
                gender_label = "invalid"
            detail = f"W{idx}:{gender_label}/{confidence} {evidence}".strip()
            details.append(detail)

        summary = (
            "vote_summary: "
            f"male=H{counts['male']['high']}/L{counts['male']['low']}, "
            f"female=H{counts['female']['high']}/L{counts['female']['low']}, "
            f"unknown=H{counts['unknown']['high']}/L{counts['unknown']['low']}, "
            f"invalid={counts['invalid']}"
        )
        if details == []:
            return summary
        return summary + "\n" + "\n".join(details)

    def pick_gender_vote_result(
        self,
        entry: dict[str, str | int | list[str]],
        results: list[dict[str, str]],
    ) -> dict[str, str] | None:
        if results == []:
            return None
        labels = self.get_gender_labels()
        normalized: list[dict[str, str]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            gender = self.map_gender_label(result.get("gender", ""))
            confidence = result.get("confidence", "")
            evidence = result.get("evidence", "")
            context = result.get("context", "")
            if gender == labels.get("unknown"):
                confidence = "low"
            normalized.append(
                {
                    "src": result.get("src", "") if result.get("src", "") != "" else entry.get("src", ""),
                    "gender": gender,
                    "confidence": confidence,
                    "evidence": evidence,
                    "context": context,
                }
            )

        if normalized == []:
            return None

        high_male = [
            result for result in normalized
            if result.get("confidence") == "high" and result.get("gender") == labels.get("male")
        ]
        high_female = [
            result for result in normalized
            if result.get("confidence") == "high" and result.get("gender") == labels.get("female")
        ]
        conflict = high_male != [] and high_female != []
        vote_summary = self.build_gender_vote_summary(normalized, labels)

        selected: dict[str, str] | None = None
        if conflict == True:
            candidates = high_male + high_female
            selected = max(candidates, key = self.score_gender_result, default = None)
            if selected is None:
                selected = normalized[0]
            selected = selected.copy()
            selected["gender"] = labels.get("unknown")
            selected["confidence"] = "low"
            selected["vote_conflict"] = True
        else:
            if high_male != [] or high_female != []:
                selected = max(high_male + high_female, key = self.score_gender_result)
            else:
                low_candidates = [
                    result for result in normalized
                    if result.get("gender") in (labels.get("male"), labels.get("female"), labels.get("unknown"))
                ]
                counts = {"male": 0, "female": 0, "unknown": 0}
                for result in low_candidates:
                    if result.get("gender") == labels.get("male"):
                        counts["male"] += 1
                    elif result.get("gender") == labels.get("female"):
                        counts["female"] += 1
                    elif result.get("gender") == labels.get("unknown"):
                        counts["unknown"] += 1
                max_count = max(counts.values()) if counts != {} else 0
                if max_count <= 0:
                    selected = max(normalized, key = self.score_gender_result)
                else:
                    top = [key for key, value in counts.items() if value == max_count]
                    if len(top) != 1:
                        chosen = "unknown"
                    else:
                        chosen = top[0]
                    if chosen == "male":
                        pool = [result for result in low_candidates if result.get("gender") == labels.get("male")]
                    elif chosen == "female":
                        pool = [result for result in low_candidates if result.get("gender") == labels.get("female")]
                    else:
                        pool = [result for result in low_candidates if result.get("gender") == labels.get("unknown")]
                    if pool == []:
                        pool = low_candidates
                    if pool == []:
                        selected = max(normalized, key = self.score_gender_result)
                    else:
                        selected = max(pool, key = self.score_gender_result)

        if selected is None:
            return None

        selected = selected.copy()
        if vote_summary != "":
            selected["vote_summary"] = vote_summary
        if conflict == True and selected.get("vote_conflict") is not True:
            selected["vote_conflict"] = True
        return selected

    def is_snippet_too_similar(self, text: str, selected: list[dict[str, str | int]], threshold: float) -> bool:
        for item in selected:
            other = str(item.get("text", ""))
            if other == "":
                continue
            if TextHelper.check_similarity_by_jaccard(text, other) >= threshold:
                return True
        return False

    def is_snippet_too_close(self, snippet: dict[str, str | int], selected: list[dict[str, str | int]], min_distance: int) -> bool:
        try:
            order = int(snippet.get("order", 0))
        except Exception:
            order = 0
        file_path = str(snippet.get("file", ""))

        for item in selected:
            if file_path != "" and file_path != str(item.get("file", "")):
                continue
            try:
                other = int(item.get("order", 0))
            except Exception:
                other = 0
            if abs(order - other) <= min_distance:
                return True
        return False

    def get_gender_labels(self) -> dict[str, str]:
        if self.config.target_language == BaseLanguage.Enum.ZH:
            return {
                "male": "男性人名",
                "female": "女性人名",
                "unknown": "未知性别人名",
            }
        else:
            return {
                "male": "Male Name",
                "female": "Female Name",
                "unknown": "Name of Unknown Gender",
            }

    def is_surname(self, info: str) -> bool:
        if not isinstance(info, str):
            return False
        text = info.strip().lower()
        if text == "":
            return False
        if "姓氏" in text:
            return True
        return "surname" in text or "family name" in text

    def is_person_name(self, info: str) -> bool:
        if not isinstance(info, str):
            return False
        if self.is_surname(info) == True:
            return False
        text = info.strip().lower()
        return "人名" in text or "name" in text

    def normalize_info_category(self, info: str) -> str:
        if not isinstance(info, str):
            return ""
        text = info.strip().lower()
        if text == "":
            return ""
        if "姓" in text or "surname" in text or "family name" in text:
            return "surname"
        if "人名" in text or "name" in text:
            return "person"
        if "地名" in text or "place" in text or "location" in text or "city" in text or "country" in text or "town" in text:
            return "place"
        if "组织" in text or "organisation" in text or "organization" in text or "guild" in text or "団" in text or "團" in text:
            return "org"
        if "家族" in text or "family" in text or "clan" in text:
            return "family"
        if "物品" in text or "道具" in text or "item" in text:
            return "item"
        if "生物" in text or "种族" in text or "種族" in text or "creature" in text or "species" in text:
            return "creature"
        if "其他" in text or "other" in text:
            return "other"
        return "other"

    def get_info_categories(self, entry: dict[str, str | int | list[str]]) -> set[str]:
        choices = entry.get("info_choices")
        if isinstance(choices, set):
            items = choices
        elif isinstance(choices, list):
            items = choices
        else:
            items = [entry.get("info", "")]

        categories: set[str] = set()
        for info in items:
            category = self.normalize_info_category(info if isinstance(info, str) else str(info))
            if category != "" and category != "other":
                categories.add(category)

        return categories

    def has_type_conflict(self, entry: dict[str, str | int | list[str]]) -> bool:
        return len(self.get_info_categories(entry)) >= 2

    def format_info_choices(self, entry: dict[str, str | int | list[str]]) -> str:
        choices = entry.get("info_choices")
        if isinstance(choices, set):
            values = choices
        elif isinstance(choices, list):
            values = choices
        else:
            values = [entry.get("info", "")]
        clean = sorted({str(v).strip() for v in values if str(v).strip() != ""})
        return ", ".join(clean)

    def is_unknown_gender(self, info: str) -> bool:
        if not isinstance(info, str):
            return False
        labels = self.get_gender_labels()
        text = info.strip().lower()
        if text == labels.get("unknown", "").lower():
            return True
        return ("unknown" in text and "name" in text) or ("未知" in text)

    def map_gender_label(self, gender: str) -> str:
        labels = self.get_gender_labels()
        if gender == labels.get("male") or gender == "男性人名":
            return labels.get("male")
        if gender == labels.get("female") or gender == "女性人名":
            return labels.get("female")
        if gender == labels.get("unknown"):
            return labels.get("unknown")

        text = str(gender).strip().lower()
        if (
            "unknown" in text
            or "uncertain" in text
            or "unsure" in text
            or "not sure" in text
            or "不确定" in text
            or "未知" in text
        ):
            return labels.get("unknown")
        if "male" in text or "男" in text:
            return labels.get("male")
        if "female" in text or "女" in text:
            return labels.get("female")

        return ""

    def is_evidence_valid(self, evidence: str) -> bool:
        if not isinstance(evidence, str):
            return False
        text = evidence.strip()
        if text == "":
            return False
        return re.search(r"S\d{3}", text, flags = re.IGNORECASE) is not None

    def is_validator_result_invalid(self, result: dict[str, str | bool] | None) -> bool:
        if not isinstance(result, dict):
            return True
        return result.get("keep") is None

    def should_retry_validator(
        self,
        entry: dict[str, str | int | list[str]],
        result: dict[str, str | bool] | None,
    ) -> bool:
        if self.is_validator_result_invalid(result) == True:
            return True
        if not isinstance(result, dict):
            return False

        reason = result.get("reason", "")
        reason_text = reason.strip().lower() if isinstance(reason, str) else str(reason).strip().lower()
        if reason_text == "":
            return False
        if any(token in reason_text for token in __class__.VALIDATOR_RETRY_KEYWORDS) == False:
            return False

        snippets = entry.get("snippets", [])
        return isinstance(snippets, list) and len(snippets) > 0

    def is_reason_insufficient(self, reason: str) -> bool:
        if not isinstance(reason, str):
            return False
        text = reason.strip().lower()
        if text == "":
            return False
        return any(token in text for token in __class__.VALIDATOR_RETRY_KEYWORDS)

    def should_review_insufficient(self, entry: dict[str, str | int | list[str]]) -> bool:
        threshold = max(0, int(self.config.multi_agent_review_high_freq_min_count))
        if threshold <= 0:
            return False
        count = entry.get("count", 0)
        try:
            count_value = int(count)
        except Exception:
            count_value = 0
        if count_value < threshold:
            return False
        reason = entry.get("validator_reason", "")
        return self.is_reason_insufficient(reason)

    def is_gender_result_invalid(self, result: dict[str, str] | None) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("gender", "") == "" or result.get("confidence", "") == "":
            return True
        if self.is_evidence_valid(result.get("evidence", "")) == False:
            return True
        return False

    def is_translator_result_invalid(self, result: dict[str, str] | None) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("dst", "") == "":
            return True
        return False

    def is_review_arbiter_result_invalid(self, result: dict[str, str | bool] | None) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("keep") is None:
            return True
        if result.get("confidence", "") == "":
            return True
        keep = result.get("keep") is True
        if keep == True and result.get("info", "") == "":
            return True
        if self.is_evidence_valid(result.get("evidence", "")) == False:
            return True
        return False

    def normalize_translation(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.strip()

    def pick_best_translation(
        self,
        entry: dict[str, str | int | list[str]],
        candidates: list[str],
        existing: str,
    ) -> str:
        if candidates == []:
            return existing

        src = str(entry.get("src", ""))
        dst_choices = entry.get("dst_choices", set())
        if isinstance(dst_choices, set):
            choice_set = {self.normalize_translation(v) for v in dst_choices if isinstance(v, str) and v != ""}
        elif isinstance(dst_choices, list):
            choice_set = {self.normalize_translation(v) for v in dst_choices if isinstance(v, str) and v != ""}
        else:
            choice_set = set()

        scored: list[tuple[int, int, str]] = []
        for candidate in candidates:
            candidate = self.normalize_translation(candidate)
            if candidate == "":
                continue
            score = 0
            if existing != "" and candidate == existing:
                score += 3
            if candidate in choice_set:
                score += 2
            if candidate == src:
                score -= 3
            if TextHelper.get_display_lenght(candidate) > 32:
                score -= 1
            scored.append((score, TextHelper.get_display_lenght(candidate), candidate))

        if scored == []:
            return existing

        scored.sort(key = lambda x: (-x[0], x[1], x[2]))
        return scored[0][2]

    def apply_multi_agent_pipeline(self, glossary: list[dict[str, str | int | list[str]]]) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        if glossary == []:
            return glossary, []

        prompt_builder = PromptBuilder(self.config)
        prompt_language, _, _ = prompt_builder.resolve_prompt_language()
        max_workers, rpm_threshold = self.initialize_max_workers()

        title_review_entries: list[dict[str, str]] = []
        if self.config.multi_agent_title_filter_enable == True:
            glossary, title_review_entries = self.apply_title_filter(glossary, prompt_language)

        self.reset_agent_usage()
        glossary, validator_review_entries = self.run_validator(glossary, prompt_builder, prompt_language, max_workers, rpm_threshold)
        self.flush_agent_usage()
        self.reset_agent_usage()
        glossary, gender_review_entries = self.run_gender(glossary, prompt_builder, prompt_language, max_workers, rpm_threshold)
        self.flush_agent_usage()
        translator_review_entries: list[dict[str, str]] = []
        if self.config.multi_agent_translate_post == True:
            self.reset_agent_usage()
            glossary, translator_review_entries = self.run_translator(glossary, prompt_builder, prompt_language, max_workers, rpm_threshold)
            self.flush_agent_usage()
        extra_review_entries = self.collect_review_entries(glossary, prompt_language)
        review_entries = self.merge_review_entries(
            title_review_entries,
            validator_review_entries,
            gender_review_entries,
            translator_review_entries,
            extra_review_entries,
        )
        if self.config.multi_agent_review_arbitrate == True:
            self.reset_agent_usage()
            glossary, arbiter_review_entries = self.run_review_arbiter(
                glossary,
                review_entries,
                prompt_builder,
                prompt_language,
                max_workers,
                rpm_threshold,
            )
            self.flush_agent_usage()
            review_entries = self.merge_review_entries(review_entries, arbiter_review_entries)

        for entry in glossary:
            entry.pop("snippets", None)
            entry.pop("validator_reason", None)
            entry.pop("translation_candidates", None)
            entry.pop("translation_conflict", None)

        return glossary, review_entries

    def apply_title_filter(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        prompt_language: BaseLanguage.Enum,
    ) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        if glossary == []:
            return glossary, []
        if self.config.multi_agent_title_filter_enable != True:
            return glossary, []

        budget = max(0, int(self.config.multi_agent_context_budget))
        kept: list[dict[str, str | int | list[str]]] = []
        review_entries: list[dict[str, str]] = []

        for entry in glossary:
            if TitleFilter.filter(str(entry.get("src", ""))):
                context = self.format_snippet_context(entry.get("snippets", []), budget, prompt_language)
                review_entries.append(
                    self.build_validator_review_entry(
                        entry,
                        prompt_language,
                        "title_filtered",
                        context,
                    )
                )
            kept.append(entry)

        return kept, review_entries

    def collect_review_entries(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        prompt_language: BaseLanguage.Enum,
    ) -> list[dict[str, str]]:
        if glossary == []:
            return []

        budget = max(0, int(self.config.multi_agent_context_budget))
        review_entries: list[dict[str, str]] = []

        for entry in glossary:
            if self.has_type_conflict(entry):
                context = self.format_snippet_context(entry.get("snippets", []), budget, prompt_language)
                choices = self.format_info_choices(entry)
                if choices != "":
                    context = f"info_choices={choices}\n\n{context}"
                review_entries.append(
                    self.build_validator_review_entry(
                        entry,
                        prompt_language,
                        "type_conflict",
                        context,
                    )
                )

            if self.should_review_insufficient(entry):
                context = self.format_snippet_context(entry.get("snippets", []), budget, prompt_language)
                reason = entry.get("validator_reason", "")
                if isinstance(reason, str) and reason.strip() != "":
                    context = f"validator_reason={reason.strip()}\n\n{context}"
                review_entries.append(
                    self.build_validator_review_entry(
                        entry,
                        prompt_language,
                        "insufficient_evidence",
                        context,
                    )
                )

        return review_entries

    def reset_agent_usage(self) -> None:
        with self.agent_usage_lock:
            self.agent_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def record_agent_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        if input_tokens == 0 and output_tokens == 0:
            return None

        with self.agent_usage_lock:
            self.agent_usage["input_tokens"] = self.agent_usage.get("input_tokens", 0) + input_tokens
            self.agent_usage["output_tokens"] = self.agent_usage.get("output_tokens", 0) + output_tokens

    def flush_agent_usage(self) -> None:
        with self.agent_usage_lock:
            input_tokens = self.agent_usage.get("input_tokens", 0)
            output_tokens = self.agent_usage.get("output_tokens", 0)
            self.agent_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

        if input_tokens == 0 and output_tokens == 0:
            return None

        if not isinstance(getattr(self, "extras", None), dict):
            return None

        with self.lock:
            self.extras["total_tokens"] = self.extras.get("total_tokens", 0) + input_tokens + output_tokens
            self.extras["total_output_tokens"] = self.extras.get("total_output_tokens", 0) + output_tokens
            self.extras["time"] = time.time() - self.extras.get("start_time", 0)

        self.cache_manager.get_project().set_extras(self.extras)
        self.emit(Base.Event.NER_ANALYZER_UPDATE, self.extras)

    def emit_stage_progress(self, stage: str, progress: int, total: int) -> None:
        if not isinstance(getattr(self, "extras", None), dict):
            return None

        with self.lock:
            self.extras["stage"] = stage
            self.extras["stage_progress"] = progress
            self.extras["stage_total"] = total
            self.extras["time"] = time.time() - self.extras.get("start_time", 0)

        self.emit(Base.Event.NER_ANALYZER_UPDATE, self.extras)

    def log_agent_response(
        self,
        agent: str,
        src: str,
        response_think: str | None,
        response_result: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        logs: list[str] = []
        if src:
            logs.append(f"[{agent}] {Localizer.get().ner_output_log_src}{src}")
        if input_tokens or output_tokens:
            logs.append(f"[{agent}] tokens: in={int(input_tokens or 0)} out={int(output_tokens or 0)}")
        if isinstance(response_think, str) and response_think != "":
            logs.append(Localizer.get().engine_response_think + "\n" + response_think)
        if isinstance(response_result, str) and response_result != "":
            logs.append(Localizer.get().engine_response_result + "\n" + response_result)

        if logs == []:
            return None

        self.info(
            "\n" + "\n\n".join(logs) + "\n",
            file = True,
            console = LogManager.get().is_expert_mode(),
        )

    def run_validator(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        max_workers: int,
        rpm_threshold: int,
    ) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        if glossary == []:
            return glossary, []

        task_limiter = TaskLimiter(rps = max_workers, rpm = rpm_threshold)
        results: dict[str, dict[str, str | bool]] = {}

        self.emit_stage_progress("validating", 0, len(glossary))
        with ProgressBar(transient = True) as progress:
            pid = progress.new()
            with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers, thread_name_prefix = Engine.TASK_PREFIX) as executor:
                futures = []
                for entry in glossary:
                    task_limiter.wait()
                    futures.append(executor.submit(self.request_validator, entry, prompt_builder, prompt_language))

                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and result.get("src"):
                        results[result.get("src")] = result
                    completed_count += 1
                    progress.update(pid, advance = 1, total = len(futures))
                    self.emit_stage_progress("validating", completed_count, len(futures))

        kept: list[dict[str, str | int | list[str]]] = []
        review_entries: list[dict[str, str]] = []
        for entry in glossary:
            decision = results.get(entry.get("src"))
            if isinstance(decision, dict):
                if decision.get("reason"):
                    entry["validator_reason"] = decision.get("reason")
                if decision.get("keep") is False:
                    continue
                if decision.get("keep") is None:
                    review_entries.append(
                        self.build_validator_review_entry(
                            entry,
                            prompt_language,
                            str(decision.get("reason", "")),
                            str(decision.get("context", "")),
                        )
                    )
                    kept.append(entry)
                    continue
                kept.append(entry)
            else:
                review_entries.append(
                    self.build_validator_review_entry(
                        entry,
                        prompt_language,
                        "missing",
                        "",
                    )
                )
                kept.append(entry)

        return kept, review_entries

    def run_gender(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        max_workers: int,
        rpm_threshold: int,
    ) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        candidates = [
            entry for entry in glossary
            if self.is_person_name(entry.get("info", ""))
        ]
        if candidates == []:
            return glossary, []

        task_limiter = TaskLimiter(rps = max_workers, rpm = rpm_threshold)
        results: dict[str, dict[str, str]] = {}

        self.emit_stage_progress("gendering", 0, len(candidates))
        with ProgressBar(transient = True) as progress:
            pid = progress.new()
            with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers, thread_name_prefix = Engine.TASK_PREFIX) as executor:
                futures = []
                for entry in candidates:
                    task_limiter.wait()
                    futures.append(executor.submit(self.request_gender, entry, prompt_builder, prompt_language))

                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and result.get("src"):
                        results[result.get("src")] = result
                    completed_count += 1
                    progress.update(pid, advance = 1, total = len(futures))
                    self.emit_stage_progress("gendering", completed_count, len(futures))

        review_entries: list[dict[str, str]] = []
        labels = self.get_gender_labels()
        min_high_count = max(0, int(self.config.multi_agent_gender_high_confidence_min_count))

        kept: list[dict[str, str | int | list[str]]] = []
        for entry in glossary:
            if self.is_person_name(entry.get("info", "")) != True:
                kept.append(entry)
                continue

            result = results.get(entry.get("src"))
            if result is None:
                gender = labels.get("male")
                confidence = "low"
                evidence = ""
                context = ""
                review_reason = "invalid_output"
            else:
                gender = self.map_gender_label(result.get("gender", ""))
                confidence = result.get("confidence", "")
                evidence = result.get("evidence", "")
                context = result.get("context", "")
                vote_summary = result.get("vote_summary", "")
                if isinstance(vote_summary, str) and vote_summary != "":
                    context = vote_summary if context == "" else f"{vote_summary}\n\n{context}"
                review_reason = ""
                evidence_valid = self.is_evidence_valid(evidence)
                vote_conflict = result.get("vote_conflict") is True
                gender_is_unknown = gender == labels.get("unknown")

                if gender == "":
                    gender = labels.get("male")
                    confidence = "low"
                    review_reason = "invalid_output"
                elif vote_conflict:
                    confidence = "low"
                    review_reason = "gender_conflict"
                elif gender_is_unknown:
                    confidence = "low"
                elif confidence == "":
                    confidence = "low"
                    review_reason = "invalid_output"

                if evidence_valid == False:
                    confidence = "low"
                    if review_reason == "":
                        review_reason = "invalid_evidence"

            entry["info"] = gender
            info_choices = entry.get("info_choices")
            if isinstance(info_choices, set):
                info_choices.add(gender)

            if confidence != "high":
                if review_reason == "":
                    review_reason = "low_confidence"
                count = entry.get("count", 0)
                try:
                    count_value = int(count)
                except Exception:
                    count_value = 0
                if min_high_count > 0 and count_value >= min_high_count:
                    review_reason = "low_confidence_high_count"
                review_entries.append(
                    self.build_review_entry(entry, prompt_language, review_reason, evidence, context)
                )

            kept.append(entry)

        return kept, review_entries

    def run_translator(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        max_workers: int,
        rpm_threshold: int,
    ) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        if glossary == []:
            return glossary, []

        task_limiter = TaskLimiter(rps = max_workers, rpm = rpm_threshold)
        results: dict[str, dict[str, str]] = {}

        self.emit_stage_progress("translating", 0, len(glossary))
        with ProgressBar(transient = True) as progress:
            pid = progress.new()
            with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers, thread_name_prefix = Engine.TASK_PREFIX) as executor:
                futures = []
                for entry in glossary:
                    task_limiter.wait()
                    futures.append(executor.submit(self.request_translator, entry, prompt_builder, prompt_language))

                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and result.get("src"):
                        results[result.get("src")] = result
                    completed_count += 1
                    progress.update(pid, advance = 1, total = len(futures))
                    self.emit_stage_progress("translating", completed_count, len(futures))

        review_entries: list[dict[str, str]] = []
        for entry in glossary:
            result = results.get(entry.get("src"))
            if result is None:
                review_entries.append(
                    self.build_translation_review_entry(
                        entry,
                        prompt_language,
                        "missing",
                        "",
                        "",
                        "",
                    )
                )
                continue

            translated = self.normalize_translation(result.get("dst", ""))
            if translated == "":
                review_entries.append(
                    self.build_translation_review_entry(
                        entry,
                        prompt_language,
                        "invalid_output",
                        "",
                        result.get("context", ""),
                        "",
                    )
                )
                continue

            translated = self.convert_chinese_character_form(translated)
            existing = self.normalize_translation(entry.get("dst", ""))
            conflict = bool(result.get("translation_conflict"))
            candidates = result.get("translation_candidates")

            merged_candidates: list[str] = []
            if isinstance(candidates, list):
                merged_candidates.extend([self.normalize_translation(v) for v in candidates if isinstance(v, str) and v.strip() != ""])
            if existing != "":
                merged_candidates.append(existing)
            if translated != "":
                merged_candidates.append(translated)
            merged_candidates = list(dict.fromkeys([v for v in merged_candidates if v != ""]))

            chosen = self.pick_best_translation(entry, merged_candidates, existing)
            if chosen == "":
                chosen = translated if translated != "" else existing
            entry["dst"] = chosen

            if conflict == True or (existing != "" and translated != "" and existing != translated) or len(merged_candidates) > 1:
                entry["translation_conflict"] = True
                entry["translation_candidates"] = merged_candidates
                review_context = result.get("context", "")
                if chosen not in ("", translated, existing):
                    review_context = f"picked={chosen}" if review_context == "" else f"picked={chosen}\n\n{review_context}"
                review_entries.append(
                    self.build_translation_review_entry(
                        entry,
                        prompt_language,
                        "conflict",
                        translated,
                        review_context,
                        existing,
                    )
                )

            dst_choices = entry.get("dst_choices")
            if isinstance(dst_choices, set):
                if chosen != "":
                    dst_choices.add(chosen)
                if translated != "" and translated != chosen:
                    dst_choices.add(translated)
                if existing != "" and existing != chosen:
                    dst_choices.add(existing)
                dst_choices.discard("")

        return glossary, review_entries

    def run_review_arbiter(
        self,
        glossary: list[dict[str, str | int | list[str]]],
        review_entries: list[dict[str, str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        max_workers: int,
        rpm_threshold: int,
    ) -> tuple[list[dict[str, str | int | list[str]]], list[dict[str, str]]]:
        if glossary == [] or review_entries == []:
            return glossary, []
        if self.config.multi_agent_review_arbitrate != True:
            return glossary, []

        review_reason_map = self.build_review_reason_map(review_entries)
        if review_reason_map == {}:
            return glossary, []

        glossary_map = {entry.get("src", ""): entry for entry in glossary if entry.get("src", "") != ""}
        candidates = [glossary_map.get(src) for src in review_reason_map.keys() if glossary_map.get(src) is not None]
        if candidates == []:
            return glossary, []

        task_limiter = TaskLimiter(rps = max_workers, rpm = rpm_threshold)
        results: dict[str, dict[str, str | bool]] = {}

        self.emit_stage_progress("arbitrating", 0, len(candidates))
        with ProgressBar(transient = True) as progress:
            pid = progress.new()
            with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers, thread_name_prefix = Engine.TASK_PREFIX) as executor:
                futures = []
                for entry in candidates:
                    task_limiter.wait()
                    reason = review_reason_map.get(entry.get("src", ""), "")
                    futures.append(executor.submit(self.request_review_arbiter, entry, reason, prompt_builder, prompt_language))

                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, dict) and result.get("src"):
                        results[result.get("src")] = result
                    completed_count += 1
                    progress.update(pid, advance = 1, total = len(futures))
                    self.emit_stage_progress("arbitrating", completed_count, len(futures))

        apply_to_main = self.config.multi_agent_review_arbitrate_apply == True
        arbiter_review_entries: list[dict[str, str]] = []
        for entry in candidates:
            result = results.get(entry.get("src"))
            if result is None:
                arbiter_review_entries.append(
                    self.build_arbiter_review_entry(entry, prompt_language, "arbiter_missing", "", "")
                )
                continue

            if self.is_review_arbiter_result_invalid(result):
                arbiter_review_entries.append(
                    self.build_arbiter_review_entry(entry, prompt_language, "arbiter_invalid", result.get("evidence", ""), result.get("context", ""))
                )
                continue

            keep = result.get("keep") is True
            confidence = result.get("confidence", "")
            evidence = result.get("evidence", "")
            context = result.get("context", "")
            arbiter_reason = result.get("reason", "")
            if isinstance(arbiter_reason, str) and arbiter_reason.strip() != "":
                context = f"arbiter_reason={arbiter_reason.strip()}\n\n{context}" if context != "" else f"arbiter_reason={arbiter_reason.strip()}"

            if keep == False:
                arbiter_review_entries.append(
                    self.build_arbiter_review_entry(entry, prompt_language, "arbiter_keep_false", evidence, context)
                )
            elif confidence != "high":
                arbiter_review_entries.append(
                    self.build_arbiter_review_entry(entry, prompt_language, "arbiter_low_confidence", evidence, context)
                )

            if apply_to_main == True and keep == True and confidence == "high":
                info = result.get("info", "")
                info = info if isinstance(info, str) else str(info)
                info = info.strip()
                mapped = self.map_gender_label(info)
                if mapped != "":
                    info = mapped
                if info != "":
                    entry["info"] = info
                    info_choices = entry.get("info_choices")
                    if isinstance(info_choices, set):
                        info_choices.add(info)

        return glossary, arbiter_review_entries

    def request_review_arbiter(
        self,
        entry: dict[str, str | int | list[str]],
        review_reason: str,
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
    ) -> dict[str, str | bool] | None:
        short_budget = max(0, int(self.config.multi_agent_context_budget))
        long_budget = max(0, int(self.config.multi_agent_context_budget_long))

        short_result = self.request_review_arbiter_once(entry, review_reason, prompt_builder, prompt_language, short_budget)
        needs_retry = self.is_review_arbiter_result_invalid(short_result)
        if needs_retry == False and short_result is not None and short_result.get("confidence") != "high":
            needs_retry = True

        if needs_retry == True:
            retry_budget = long_budget if long_budget > short_budget else short_budget
            long_result = self.request_review_arbiter_once(entry, review_reason, prompt_builder, prompt_language, retry_budget)
            return self.pick_review_arbiter_result(short_result, long_result)

        return short_result

    def request_review_arbiter_once(
        self,
        entry: dict[str, str | int | list[str]],
        review_reason: str,
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        budget: int,
    ) -> dict[str, str | bool] | None:
        context = self.format_snippet_context(
            entry.get("snippets", []),
            budget,
            prompt_language,
        )
        info_choices = self.format_info_choices(entry)
        prompt = prompt_builder.build_agent_prompt(
            "arbiter",
            {
                "src": entry.get("src", ""),
                "type_hint": entry.get("info", ""),
                "info_choices": info_choices,
                "review_reason": review_reason,
                "context": context,
            },
        )
        if prompt == "":
            return None

        requester = TaskRequester(self.config, self.platform)
        skip, response_think, response_result, input_tokens, output_tokens = requester.request([
            {
                "role": "user",
                "content": prompt,
            }
        ])
        self.record_agent_usage(input_tokens, output_tokens)
        if skip == False and response_result is not None:
            self.log_agent_response("arbiter", str(entry.get("src", "")), response_think, response_result, input_tokens, output_tokens)
        if skip == True or response_result is None:
            self.warning(f"[arbiter] request_failed src={entry.get('src', '')}")
            return {
                "src": entry.get("src", ""),
                "keep": None,
                "info": "",
                "confidence": "",
                "evidence": "",
                "context": context,
            }

        results = ResponseDecoder().decode_arbiter(response_result)
        if results != []:
            result = results[0]
            if result.get("src", "") == "":
                result["src"] = entry.get("src", "")
            result["context"] = context
            return result

        return {
            "src": entry.get("src", ""),
            "keep": None,
            "info": "",
            "confidence": "",
            "evidence": "",
            "context": context,
        }

    def pick_review_arbiter_result(
        self,
        short_result: dict[str, str | bool] | None,
        long_result: dict[str, str | bool] | None,
    ) -> dict[str, str | bool] | None:
        if long_result is None:
            return short_result
        if short_result is None:
            return long_result

        short_invalid = self.is_review_arbiter_result_invalid(short_result)
        long_invalid = self.is_review_arbiter_result_invalid(long_result)
        if short_invalid == True and long_invalid == False:
            return long_result
        if long_invalid == True and short_invalid == False:
            return short_result
        if short_invalid == True and long_invalid == True:
            return long_result

        short_confidence = short_result.get("confidence", "")
        long_confidence = long_result.get("confidence", "")
        if short_confidence != "high" and long_confidence == "high":
            return long_result
        if short_confidence == "high" and long_confidence != "high":
            return short_result

        short_keep = short_result.get("keep") is True
        long_keep = long_result.get("keep") is True
        if short_keep == False and long_keep == True:
            return long_result
        if short_keep == True and long_keep == False:
            return short_result

        return short_result

    def request_translator(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
    ) -> dict[str, str] | None:
        short_budget = max(0, int(self.config.multi_agent_context_budget))
        long_budget = max(0, int(self.config.multi_agent_context_budget_long))

        short_result = self.request_translator_once(entry, prompt_builder, prompt_language, short_budget)
        needs_retry = self.is_translator_result_invalid(short_result)
        if needs_retry == False:
            existing = self.normalize_translation(entry.get("dst", ""))
            proposed = self.normalize_translation(short_result.get("dst", "") if isinstance(short_result, dict) else "")
            if existing != "" and proposed != "" and existing != proposed:
                needs_retry = True

        if needs_retry == True:
            retry_budget = long_budget if long_budget > short_budget else short_budget
            long_result = self.request_translator_once(entry, prompt_builder, prompt_language, retry_budget)
            return self.pick_translation_result(entry, short_result, long_result)

        return short_result

    def request_translator_once(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        budget: int,
    ) -> dict[str, str] | None:
        context = self.format_snippet_context(
            entry.get("snippets", []),
            budget,
            prompt_language,
        )
        prompt = prompt_builder.build_agent_prompt(
            "translator",
            {
                "src": entry.get("src", ""),
                "type_hint": entry.get("info", ""),
                "context": context,
            },
        )
        if prompt == "":
            return None

        requester = TaskRequester(self.config, self.platform)
        skip, response_think, response_result, input_tokens, output_tokens = requester.request([
            {
                "role": "user",
                "content": prompt,
            }
        ])
        self.record_agent_usage(input_tokens, output_tokens)
        if skip == False and response_result is not None:
            self.log_agent_response("translator", str(entry.get("src", "")), response_think, response_result, input_tokens, output_tokens)
        if skip == True or response_result is None:
            self.warning(f"[translator] request_failed src={entry.get('src', '')}")
            return {
                "src": entry.get("src", ""),
                "dst": "",
                "context": context,
            }

        results = ResponseDecoder().decode_translator(response_result)
        if results != []:
            result = results[0]
            if result.get("src", "") == "":
                result["src"] = entry.get("src", "")
            result["context"] = context
            return result

        return {
            "src": entry.get("src", ""),
            "dst": "",
            "context": context,
        }

    def pick_translation_result(
        self,
        entry: dict[str, str | int | list[str]],
        short_result: dict[str, str] | None,
        long_result: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if long_result is None:
            return short_result
        if short_result is None:
            return long_result

        short_invalid = self.is_translator_result_invalid(short_result)
        long_invalid = self.is_translator_result_invalid(long_result)
        if short_invalid == True and long_invalid == False:
            return long_result
        if long_invalid == True and short_invalid == False:
            return short_result
        if short_invalid == True and long_invalid == True:
            return long_result

        short_dst = self.normalize_translation(short_result.get("dst", ""))
        long_dst = self.normalize_translation(long_result.get("dst", ""))
        if short_dst == long_dst:
            return short_result

        existing = self.normalize_translation(entry.get("dst", ""))
        if existing in (short_dst, long_dst) and existing != "":
            return short_result if existing == short_dst else long_result

        long_result["translation_conflict"] = True
        long_result["translation_candidates"] = [short_dst, long_dst]
        return long_result

    def request_validator(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
    ) -> dict[str, str | bool] | None:
        short_budget = max(0, int(self.config.multi_agent_context_budget))
        long_budget = max(0, int(self.config.multi_agent_context_budget_long))

        short_result = self.request_validator_once(entry, prompt_builder, prompt_language, short_budget)
        if self.should_retry_validator(entry, short_result) == True:
            retry_budget = long_budget if long_budget > short_budget else short_budget
            long_result = self.request_validator_once(entry, prompt_builder, prompt_language, retry_budget)
            return self.pick_validator_result(short_result, long_result)

        return short_result

    def request_validator_once(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        budget: int,
    ) -> dict[str, str | bool] | None:
        context = self.format_snippet_context(
            entry.get("snippets", []),
            budget,
            prompt_language,
        )
        prompt = prompt_builder.build_agent_prompt(
            "validator",
            {
                "src": entry.get("src", ""),
                "type_hint": entry.get("info", ""),
                "context": context,
            },
        )
        if prompt == "":
            return None

        requester = TaskRequester(self.config, self.platform)
        skip, response_think, response_result, input_tokens, output_tokens = requester.request([
            {
                "role": "user",
                "content": prompt,
            }
        ])
        self.record_agent_usage(input_tokens, output_tokens)
        if skip == False and response_result is not None:
            self.log_agent_response("validator", str(entry.get("src", "")), response_think, response_result, input_tokens, output_tokens)
        if skip == True or response_result is None:
            self.warning(f"[validator] request_failed src={entry.get('src', '')}")
            return {
                "src": entry.get("src", ""),
                "keep": None,
                "reason": "request_failed",
                "context": context,
            }

        results = ResponseDecoder().decode_validator(response_result)
        if results != []:
            result = results[0]
            if result.get("src", "") == "":
                result["src"] = entry.get("src", "")
            result["context"] = context
            return result

        return {
            "src": entry.get("src", ""),
            "keep": None,
            "reason": "parse_failed",
            "context": context,
        }

    def pick_validator_result(
        self,
        short_result: dict[str, str | bool] | None,
        long_result: dict[str, str | bool] | None,
    ) -> dict[str, str | bool] | None:
        if long_result is None:
            return short_result
        if short_result is None:
            return long_result

        short_invalid = self.is_validator_result_invalid(short_result)
        long_invalid = self.is_validator_result_invalid(long_result)
        if short_invalid == True and long_invalid == False:
            return long_result
        if long_invalid == True and short_invalid == False:
            return short_result

        return long_result

    def request_gender_for_snippets(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        snippets: list[dict[str, str | int]] | None,
    ) -> dict[str, str] | None:
        short_budget = max(0, int(self.config.multi_agent_context_budget))
        long_budget = max(0, int(self.config.multi_agent_context_budget_long))

        result = self.request_gender_once(entry, prompt_builder, prompt_language, short_budget, snippets)
        needs_retry = self.is_gender_result_invalid(result)
        if (
            needs_retry == False
            and result is not None
            and result.get("confidence") != "high"
            and self.config.multi_agent_gender_retry_long == True
        ):
            needs_retry = True
        if needs_retry == True:
            retry_budget = long_budget if long_budget > short_budget else short_budget
            long_result = self.request_gender_once(entry, prompt_builder, prompt_language, retry_budget, snippets)
            result = self.pick_gender_result(result, long_result)

        return result

    def request_gender(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
    ) -> dict[str, str] | None:
        snippets = entry.get("snippets", [])
        if not isinstance(snippets, list):
            snippets = []
        windows = self.get_gender_vote_window_count(entry)
        if windows >= 2 and snippets != []:
            groups = self.split_snippets_for_gender_vote(snippets, windows)
            vote_results: list[dict[str, str]] = []
            for group in groups:
                result = self.request_gender_for_snippets(entry, prompt_builder, prompt_language, group)
                if result is not None:
                    vote_results.append(result)
            voted = self.pick_gender_vote_result(entry, vote_results)
            if voted is not None:
                return voted

        return self.request_gender_for_snippets(entry, prompt_builder, prompt_language, snippets)

    def request_gender_once(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_builder: PromptBuilder,
        prompt_language: BaseLanguage.Enum,
        budget: int,
        snippets: list[dict[str, str | int]] | None = None,
    ) -> dict[str, str] | None:
        if snippets is None:
            snippets = entry.get("snippets", [])
        context = self.format_snippet_context(snippets, budget, prompt_language)
        prompt = prompt_builder.build_agent_prompt(
            "gender",
            {
                "src": entry.get("src", ""),
                "context": context,
            },
        )
        if prompt == "":
            return None

        requester = TaskRequester(self.config, self.platform)
        skip, response_think, response_result, input_tokens, output_tokens = requester.request([
            {
                "role": "user",
                "content": prompt,
            }
        ])
        self.record_agent_usage(input_tokens, output_tokens)
        if skip == False and response_result is not None:
            self.log_agent_response("gender", str(entry.get("src", "")), response_think, response_result, input_tokens, output_tokens)
        if skip == True or response_result is None:
            self.warning(f"[gender] request_failed src={entry.get('src', '')}")
            return {
                "src": entry.get("src", ""),
                "gender": "",
                "confidence": "",
                "evidence": "",
                "context": context,
            }

        results = ResponseDecoder().decode_gender(response_result)
        if results != []:
            result = results[0]
            if result.get("src", "") == "":
                result["src"] = entry.get("src", "")
            result["context"] = context
            return result

        return {
            "src": entry.get("src", ""),
            "gender": "",
            "confidence": "",
            "evidence": "",
            "context": context,
        }

    def pick_gender_result(self, short_result: dict[str, str] | None, long_result: dict[str, str] | None) -> dict[str, str] | None:
        if long_result is None:
            return short_result
        if short_result is None:
            return long_result

        short_invalid = self.is_gender_result_invalid(short_result)
        long_invalid = self.is_gender_result_invalid(long_result)
        if short_invalid == True and long_invalid == False:
            return long_result
        if long_invalid == True and short_invalid == False:
            return short_result
        if short_invalid == True and long_invalid == True:
            return long_result

        short_confidence = short_result.get("confidence", "")
        long_confidence = long_result.get("confidence", "")
        if short_confidence != "high" and long_confidence == "high":
            return long_result

        short_evidence_valid = self.is_evidence_valid(short_result.get("evidence", ""))
        long_evidence_valid = self.is_evidence_valid(long_result.get("evidence", ""))
        if short_evidence_valid == False and long_evidence_valid == True:
            return long_result
        if short_evidence_valid == True and long_evidence_valid == False:
            return short_result

        return short_result

    def build_review_entry(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_language: BaseLanguage.Enum,
        reason_key: str,
        evidence: str,
        context: str,
    ) -> dict[str, str]:
        if prompt_language == BaseLanguage.Enum.ZH:
            if reason_key == "low_confidence":
                reason = "证据不足/冲突"
            elif reason_key == "low_confidence_high_count":
                reason = "高频低置信度"
            elif reason_key == "invalid_evidence":
                reason = "证据不合格"
            elif reason_key == "gender_conflict":
                reason = "性别冲突"
            else:
                reason = "模型输出异常"
            info = f"{entry.get('info', '')}（需复核：{reason}）"
        else:
            if reason_key == "low_confidence":
                reason = "insufficient/conflicting evidence"
            elif reason_key == "low_confidence_high_count":
                reason = "high-frequency low confidence"
            elif reason_key == "invalid_evidence":
                reason = "invalid evidence"
            elif reason_key == "gender_conflict":
                reason = "gender conflict"
            else:
                reason = "invalid model output"
            info = f"{entry.get('info', '')} (Review: {reason})"

        return {
            "src": entry.get("src", ""),
            "dst": entry.get("dst", ""),
            "info": info,
            "review_reason": reason,
            "review_evidence": evidence,
            "review_context": context,
        }

    def build_translation_review_entry(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_language: BaseLanguage.Enum,
        reason_key: str,
        translated: str,
        context: str,
        existing: str,
    ) -> dict[str, str]:
        reason_key = reason_key.strip().lower() if isinstance(reason_key, str) else ""
        if prompt_language == BaseLanguage.Enum.ZH:
            if reason_key == "conflict":
                reason = "翻译冲突"
            elif reason_key == "missing":
                reason = "翻译结果缺失"
            else:
                reason = "翻译输出异常"
            info = f"{entry.get('info', '')}（需复核：{reason}）"
        else:
            if reason_key == "conflict":
                reason = "translation conflict"
            elif reason_key == "missing":
                reason = "translation missing"
            else:
                reason = "invalid translation output"
            info = f"{entry.get('info', '')} (Review: {reason})"

        details: list[str] = []
        if isinstance(existing, str) and existing != "":
            details.append(f"existing={existing}")
        if isinstance(translated, str) and translated != "":
            details.append(f"translated={translated}")

        if details != []:
            detail_text = " | ".join(details)
            context = detail_text if context == "" else f"{detail_text}\n\n{context}"

        return {
            "src": entry.get("src", ""),
            "dst": entry.get("dst", ""),
            "info": info,
            "review_reason": reason,
            "review_evidence": "",
            "review_context": context,
        }

    def build_arbiter_review_entry(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_language: BaseLanguage.Enum,
        reason_key: str,
        evidence: str,
        context: str,
    ) -> dict[str, str]:
        review_entry = self.build_validator_review_entry(entry, prompt_language, reason_key, context)
        if isinstance(evidence, str) and evidence != "":
            review_entry["review_evidence"] = evidence
        return review_entry

    def build_validator_review_entry(
        self,
        entry: dict[str, str | int | list[str]],
        prompt_language: BaseLanguage.Enum,
        reason_key: str,
        context: str,
    ) -> dict[str, str]:
        reason_key = reason_key.strip().lower() if isinstance(reason_key, str) else ""
        reason_detail = ""
        if prompt_language == BaseLanguage.Enum.ZH:
            if reason_key == "request_failed":
                reason = "验证请求失败"
            elif reason_key == "parse_failed":
                reason = "验证解析失败"
            elif reason_key == "missing":
                reason = "验证结果缺失"
            elif reason_key == "title_filtered":
                reason = "称谓/头衔过滤"
            elif reason_key == "type_conflict":
                reason = "类型冲突"
            elif reason_key == "insufficient_evidence":
                reason = "证据不足"
            elif reason_key == "arbiter_keep_false":
                reason = "仲裁建议剔除"
            elif reason_key == "arbiter_low_confidence":
                reason = "仲裁低置信度"
            elif reason_key == "arbiter_invalid":
                reason = "仲裁输出异常"
            elif reason_key == "arbiter_missing":
                reason = "仲裁结果缺失"
            else:
                reason = "验证失败"
                reason_detail = reason_key
            info = f"{entry.get('info', '')}（需复核：{reason}）"
        else:
            if reason_key == "request_failed":
                reason = "validator request failed"
            elif reason_key == "parse_failed":
                reason = "validator parse failed"
            elif reason_key == "missing":
                reason = "validator result missing"
            elif reason_key == "title_filtered":
                reason = "title/honorific filtered"
            elif reason_key == "type_conflict":
                reason = "type conflict"
            elif reason_key == "insufficient_evidence":
                reason = "insufficient evidence"
            elif reason_key == "arbiter_keep_false":
                reason = "arbiter suggests removal"
            elif reason_key == "arbiter_low_confidence":
                reason = "arbiter low confidence"
            elif reason_key == "arbiter_invalid":
                reason = "arbiter output invalid"
            elif reason_key == "arbiter_missing":
                reason = "arbiter result missing"
            else:
                reason = "validator failed"
                reason_detail = reason_key
            info = f"{entry.get('info', '')} (Review: {reason})"

        review_reason = reason if reason_detail == "" else f"{reason} ({reason_detail})"

        return {
            "src": entry.get("src", ""),
            "dst": entry.get("dst", ""),
            "info": info,
            "review_reason": review_reason,
            "review_evidence": "",
            "review_context": context,
        }

    def merge_review_entries(self, *groups: list[dict[str, str]]) -> list[dict[str, str]]:
        merged: dict[str, dict[str, str]] = {}
        for group in groups:
            for entry in group:
                src = entry.get("src", "")
                if src == "":
                    continue
                if src not in merged:
                    merged[src] = entry
                    continue

                merged_entry = merged[src]
                merged_entry["review_reason"] = self.merge_review_text(
                    merged_entry.get("review_reason", ""),
                    entry.get("review_reason", ""),
                    " | ",
                )
                merged_entry["review_evidence"] = self.merge_review_text(
                    merged_entry.get("review_evidence", ""),
                    entry.get("review_evidence", ""),
                    " | ",
                )
                merged_entry["review_context"] = self.merge_review_text(
                    self.flatten_review_context(merged_entry.get("review_context", "")),
                    self.flatten_review_context(entry.get("review_context", "")),
                    "\n---\n",
                )

                if merged_entry.get("info", "") == "" and entry.get("info", "") != "":
                    merged_entry["info"] = entry.get("info", "")
                if merged_entry.get("dst", "") == "" and entry.get("dst", "") != "":
                    merged_entry["dst"] = entry.get("dst", "")
        return list(merged.values())

    def build_review_reason_map(self, review_entries: list[dict[str, str]]) -> dict[str, str]:
        reason_map: dict[str, str] = {}
        for entry in review_entries:
            src = entry.get("src", "")
            if src == "":
                continue
            reason_map[src] = self.merge_review_text(
                reason_map.get(src, ""),
                entry.get("review_reason", ""),
                " | ",
            )
        return reason_map

    def merge_review_text(self, left: str, right: str, sep: str) -> str:
        left = "" if left is None else str(left)
        right = "" if right is None else str(right)
        if left == "":
            return right
        if right == "":
            return left
        if right in left:
            return left
        return f"{left}{sep}{right}"

    def flatten_review_context(self, context: str | list[str]) -> str:
        if isinstance(context, list):
            return "\n".join([str(v) for v in context if v is not None])
        return "" if context is None else str(context)

    # 中文字型转换
    def convert_chinese_character_form(self, src: str) -> str:
        if self.config.target_language != BaseLanguage.Enum.ZH:
            return src

        if self.config.traditional_chinese_enable == True:
            return __class__.OPENCCS2T.convert(src)
        else:
            return __class__.OPENCCT2S.convert(src)

    # 翻译任务完成时
    def task_done_callback(self, future: concurrent.futures.Future, pid: TaskID, progress: ProgressBar) -> None:
        try:
            # 获取结果
            result = future.result()

            # 结果为空则跳过后续的更新步骤
            if not isinstance(result, dict) or len(result) == 0:
                return

            # 记录数据
            with self.lock:
                new = {}
                new["glossary"] = self.extras.get("glossary", []) + result.get("glossary", 0)
                new["start_time"] = self.extras.get("start_time", 0)
                new["total_line"] = self.extras.get("total_line", 0)
                new["line"] = self.extras.get("line", 0) + result.get("row_count", 0)
                new["total_tokens"] = self.extras.get("total_tokens", 0) + result.get("input_tokens", 0) + result.get("output_tokens", 0)
                new["total_output_tokens"] = self.extras.get("total_output_tokens", 0) + result.get("output_tokens", 0)
                new["time"] = time.time() - self.extras.get("start_time", 0)
                self.extras = new

            # 更新翻译进度
            self.cache_manager.get_project().set_extras(self.extras)

            # 更新翻译状态
            self.cache_manager.get_project().set_status(Base.ProjectStatus.PROCESSING)

            # 请求保存缓存文件
            self.cache_manager.require_save_to_file(self.config.output_folder)

            # 日志
            progress.update(
                pid,
                total = self.extras.get("total_line", 0),
                completed = self.extras.get("line", 0),
            )

            # 触发翻译进度更新事件
            self.emit(Base.Event.NER_ANALYZER_UPDATE, self.extras)
        except Exception as e:
            self.error(f"{Localizer.get().log_task_fail}", e)
