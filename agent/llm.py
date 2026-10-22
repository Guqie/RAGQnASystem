"""DeepSeek 的 OpenAI 兼容接口。工具调用走非流式补全，最终回答走流式输出。"""

import json
from dataclasses import dataclass, field

import urllib.error
import urllib.request


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelMessage:
    content: str = ""
    tool_calls: list = field(default_factory=list)


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _request(self, payload: dict, timeout: int):
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"大模型接口返回 {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接大模型接口：{exc.reason}") from exc

    def _payload(self, messages, temperature, max_tokens, stream, model):
        chosen = model or self.model
        payload = {
            "model": chosen,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if "reasoner" not in chosen:
            payload["temperature"] = temperature
        return payload

    def complete(self, messages, temperature: float = 0.2, max_tokens: int = 800, model: str | None = None) -> str:
        payload = self._payload(messages, temperature, max_tokens, False, model)
        with self._request(payload, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"].get("content") or ""

    def complete_with_tools(self, messages, tools, temperature: float = 0.2, max_tokens: int = 800, model: str | None = None):
        payload = self._payload(messages, temperature, max_tokens, False, model)
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        with self._request(payload, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            calls.append(
                ToolCall(
                    id=raw.get("id") or "",
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "{}",
                )
            )
        return ModelMessage(content=message.get("content") or "", tool_calls=calls)

    def stream(self, messages, temperature: float = 0.6, max_tokens: int = 2000, model: str | None = None):
        payload = self._payload(messages, temperature, max_tokens, True, model)
        with self._request(payload, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield content
