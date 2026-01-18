# KeywordGacha（Fork）

使用大模型从 `小说 / 字幕 / 游戏文本` 中自动抽取术语表，并通过多 Agent 流水线进行 **去噪、性别补全、翻译后置与复核输出**。

> 本仓库为个人 fork，工作流已与原版不同；原版项目与文档请见：https://github.com/neavo/KeywordGacha

## 你会得到什么

- **主术语表**：`output/output.xlsx`（同时生成 `output/output.json`、`output/output_detail.txt`）
  - 字段：`src`（原文）/ `dst`（译文）/ `info`（类型或性别）/ `count`（出现次数）
  - 可选：开启“输出候选数据”后，会额外输出 `dst_choices`、`info_choices`
  - 可选：开启“输出 KVJSON 文件”后，会额外生成 `output/output_kv.json`（`{src: dst}`）
- **复核表（可选）**：`output/output_review.xlsx`（同时生成 `output/output_review.json`、`output/output_review_detail.txt`）
  - 收录：低置信度、证据不足、类型冲突、翻译候选冲突、仲裁不一致等条目，方便人工快速筛查
  - 可选：开启“复核分表输出”后，会按类型拆分为 `output_review_<type>.*`

## 核心改动：多 Agent 术语表流程

相比原版“全文切片直接产出术语表”，本 fork 增加了多阶段净化流程（会显著增加调用量，但更利于人工校对）：

1. **抽取器（Extractor）**：高召回率抓取候选术语（默认不强制性别、不要求完美翻译）。
2. **验证器（Validator）**：将候选词回填到命中上下文片段，剔除泛词/头衔/残片/普通名词等噪声。
3. **性别判定（Gender）**：仅对人名，根据多窗口上下文投票与证据引用，收敛为 `男性人名 / 女性人名`（无法判断则标记低置信度并进入复核）。
4. **翻译器（Translator，可选）**：在验证与性别补全之后再翻译术语，减少“翻译了垃圾词”的浪费。
5. **复核仲裁（Arbiter，可选）**：对复核条目再跑一遍仲裁；高置信度结果可回填主表。

上下文采用 **Target-Anchor Snippets**：以术语命中行为锚点，取 `±N` 行生成片段，标记 `【TARGET】…【/TARGET】` 并编号为 `[S001]`、`[S002]`……，要求模型在 `evidence` 中引用片段编号。

## 支持的输入格式

启动任务时会递归读取 `input_folder`（或直接指定单个文件）内的内容：

- `字幕`：`.srt` / `.ass`
- `电子书/纯文本`：`.txt` / `.epub`
- `Markdown`：`.md`
- `游戏文本`：`.rpy` / `.trans` / `.xlsx` / `.json`（兼容多种提取器导出）

## 使用方式

### GUI（推荐）

1. 启动：双击 `app.exe` 或运行 `python app.py`
2. 在“接口”页配置模型：填写 `api_url`、`api_key`（可多行轮换）、`model`，可点击“测试”
3. 在“项目”页设置源/目标语言与 `input/output` 路径
4. 将文本放入 `input` 文件夹，进入“任务”页点击“开始”

完成后查看 `output` 目录；任务中断可在“任务”页选择继续（缓存保存在 `output/cache/`）。

### CLI

```bash
python app.py --cli --input_folder ./input --output_folder ./output --source_language JA --target_language ZH
```

可选参数：`--config <path>`（指定配置文件）。注意：CLI 模式下 `--input_folder/--output_folder` 需要是已存在的目录。

## 可调参数（与本 fork 相关）

在“基础设置/专家设置”中可调整：

- **回复 Token 上限**：限制单次请求的最大输出（`max_output_tokens`）
- **请求重试/退避**：失败自动重试（`request_retry_max`、`request_retry_backoff`）
- **多 Agent 流程开关**：`multi_agent_enable`、`multi_agent_translate_post`、`multi_agent_review_output`
- **上下文窗口/预算**：`multi_agent_context_window`、`multi_agent_context_budget(_long)`
- **性别判定投票/重试**：高频多窗口投票、低置信度长上下文重试
- **复核仲裁**：`multi_agent_review_arbitrate`、`multi_agent_review_arbitrate_apply`

提示词模板默认位于 `resource/prompt/{zh,en}/`，也可在“质量→自定义提示词”页面覆盖模板。

## 从源码运行 / 打包

```bash
pip install -r requirements.txt
python app.py
```

打包（Windows）：安装 `pyinstaller` 后运行 `python resource/pyinstaller.py`，输出到 `dist/KeywordGacha/`。

## 致谢与说明

- 原始项目：neavo/KeywordGacha：https://github.com/neavo/KeywordGacha
- 如用于作品发布、或涉及商业使用，请务必遵循原项目的授权与署名要求。  
