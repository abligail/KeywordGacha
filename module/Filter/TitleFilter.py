class TitleFilter:
    EXACT: tuple[str, ...] = (
        "先生",
        "老師",
        "老师",
        "師匠",
        "师匠",
        "騎士",
        "骑士",
        "隊長",
        "队长",
        "団長",
        "团长",
        "司令",
        "殿下",
        "陛下",
        "閣下",
        "阁下",
        "王",
        "女王",
        "王子",
        "王女",
        "皇帝",
        "皇后",
        "皇子",
        "皇女",
        "国王",
        "國王",
        "王妃",
        "公主",
        "圣女",
        "聖女",
        "巫女",
        "勇者",
        "魔王",
        "将军",
        "將軍",
        "大臣",
        "宰相",
        "教皇",
        "神父",
        "神官",
        "僧侣",
        "僧侶",
        "修女",
        "小姐",
        "少爷",
        "少爺",
        "夫人",
        "夫君",
        "殿",
    )
    @classmethod
    def filter(cls, src: str) -> bool:
        if not isinstance(src, str):
            return False
        text = src.strip()
        if text == "":
            return False
        if text in cls.EXACT:
            return True
        return False
