import json_repair as repair

from base.Base import Base

class ResponseDecoder(Base):

    def __init__(self) -> None:
        super().__init__()

    def _iter_json_objects(self, response: str) -> list[dict]:
        objects: list[dict] = []

        for line in response.splitlines():
            line = line.strip()
            if line == "" or line.startswith("```"):
                continue
            try:
                json_data = repair.loads(line)
            except Exception:
                continue
            if isinstance(json_data, dict):
                objects.append(json_data)
            elif isinstance(json_data, list):
                objects.extend([v for v in json_data if isinstance(v, dict)])

        if objects != []:
            return objects

        try:
            json_data = repair.loads(response)
        except Exception:
            return []

        if isinstance(json_data, dict):
            return [json_data]
        elif isinstance(json_data, list):
            return [v for v in json_data if isinstance(v, dict)]
        else:
            return []

    def _coerce_bool(self, value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "yes", "keep", "ok", "y", "t", "保留", "是", "通过"):
                return True
            if text in ("false", "no", "remove", "reject", "n", "f", "剔除", "否", "删除"):
                return False
        return None

    def _normalize_gender(self, value) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in ("male name", "male", "m", "男", "男性", "男性人名"):
            return "男性人名"
        if text in ("female name", "female", "f", "女", "女性", "女性人名"):
            return "女性人名"
        if "男" in text:
            return "男性人名"
        if "女" in text:
            return "女性人名"
        return None

    def _normalize_confidence(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return "high" if value >= 0.5 else "low"
        text = str(value).strip().lower()
        if text in ("high", "h", "高", "高置信", "高置信度"):
            return "high"
        if text in ("low", "l", "低", "低置信", "低置信度"):
            return "low"
        return ""

    # 解析文本
    def decode(self, response: str) -> tuple[list[str], list[dict[str, str]]]:
        dsts: list[str] = []
        glossary: list[dict[str, str]] = []

        # 按行解析失败时，尝试按照普通 JSON 字典进行解析
        for json_data in self._iter_json_objects(response):
            if all(v in json_data for v in ("src", "dst", "type")):
                src: str = json_data.get("src")
                dst: str = json_data.get("dst")
                type: str = json_data.get("type")
                glossary.append(
                    {
                        "src": src if isinstance(src, str) else "",
                        "dst": dst if isinstance(dst, str) else "",
                        "info": type if isinstance(type, str) else "",
                    }
                )

        # 返回默认值
        return dsts, glossary

    # 解析验证结果
    def decode_validator(self, response: str) -> list[dict[str, str | bool]]:
        results: list[dict[str, str | bool]] = []

        for json_data in self._iter_json_objects(response):
            if "src" not in json_data:
                continue
            keep_raw = json_data.get("keep", json_data.get("decision", json_data.get("result")))
            keep = self._coerce_bool(keep_raw)
            reason = json_data.get("reason", "")
            results.append(
                {
                    "src": json_data.get("src") if isinstance(json_data.get("src"), str) else "",
                    "keep": keep,
                    "reason": reason if isinstance(reason, str) else str(reason),
                }
            )

        return results

    # 解析翻译结果
    def decode_translator(self, response: str) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []

        for json_data in self._iter_json_objects(response):
            if "src" not in json_data:
                continue
            dst_raw = json_data.get("dst", json_data.get("translation", json_data.get("target")))
            results.append(
                {
                    "src": json_data.get("src") if isinstance(json_data.get("src"), str) else "",
                    "dst": dst_raw if isinstance(dst_raw, str) else ("" if dst_raw is None else str(dst_raw)),
                }
            )

        return results

    # 解析性别结果
    def decode_gender(self, response: str) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []

        for json_data in self._iter_json_objects(response):
            if "src" not in json_data:
                continue
            gender_raw = json_data.get("gender", json_data.get("sex", json_data.get("type")))
            gender = self._normalize_gender(gender_raw)
            confidence = self._normalize_confidence(json_data.get("confidence", json_data.get("conf")))
            evidence = json_data.get("evidence", "")
            results.append(
                {
                    "src": json_data.get("src") if isinstance(json_data.get("src"), str) else "",
                    "gender": gender if gender is not None else "",
                    "confidence": confidence,
                    "evidence": evidence if isinstance(evidence, str) else str(evidence),
                }
            )

        return results
