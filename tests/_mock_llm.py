"""Scripted fake OpenAI-compatible server (stdlib only) for offline tests/demos.

Run: python tests/_mock_llm.py --scenario FILE [--port N]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FINISHED = "(mock scenario finished)"


class MockLLM:
    def __init__(self, scenario, model="mock-model", n_ctx=32768,
                 host="127.0.0.1", port=0):
        if isinstance(scenario, (str, bytes)) or hasattr(scenario, "__fspath__"):
            with open(scenario, encoding="utf-8") as f:
                scenario = json.load(f)
        self.scenario = list(scenario)
        self.model = model
        self.n_ctx = n_ctx
        self.requests: list[dict] = []
        self._i = 0
        self._lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.split("?")[0] == "/v1/models":
                    self._json(200, {"object": "list", "data": [{
                        "id": mock.model, "object": "model",
                        "created": 0, "owned_by": "mock",
                        "max_model_len": mock.n_ctx,
                        "meta": {"n_ctx": mock.n_ctx, "n_ctx_train": mock.n_ctx},
                    }]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._json(400, {"error": {"message": "bad json"}})
                path = self.path.split("?")[0]
                if path == "/v1/embeddings":
                    inp = body.get("input", [])
                    if isinstance(inp, str):
                        inp = [inp]
                    vec = [0.01 * (k + 1) for k in range(8)]
                    return self._json(200, {
                        "object": "list", "model": mock.model,
                        "data": [{"object": "embedding", "index": i,
                                  "embedding": vec} for i in range(len(inp))],
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    })
                if path != "/v1/chat/completions":
                    return self._json(404, {"error": {"message": "not found"}})
                with mock._lock:
                    mock.requests.append(body)
                    if body.get("tools"):
                        if mock._i < len(mock.scenario):
                            step = mock.scenario[mock._i]
                            mock._i += 1
                        else:
                            step = {"content": FINISHED}
                    else:
                        step = {"content": "ok"}
                self._reply(body, step)

            def _reply(self, body, step):
                content = step.get("content") or ""
                reasoning = step.get("reasoning")
                tcs = [{"index": i, "id": f"call_{i}", "type": "function",
                        "function": {"name": t["name"],
                                     "arguments": json.dumps(t.get("arguments", {}))}}
                       for i, t in enumerate(step.get("tool_calls") or [])]
                finish = "tool_calls" if tcs else "stop"
                pt = sum(len(json.dumps(m)) for m in body.get("messages", [])) // 4
                ct = max(1, len(content) // 4)
                usage = {"prompt_tokens": pt, "completion_tokens": ct,
                         "total_tokens": pt + ct}
                cid = "chatcmpl-mock"
                if not body.get("stream"):
                    msg = {"role": "assistant", "content": content}
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                    if tcs:
                        msg["tool_calls"] = [{k: v for k, v in t.items() if k != "index"}
                                             for t in tcs]
                    return self._json(200, {
                        "id": cid, "object": "chat.completion", "created": 0,
                        "model": mock.model, "usage": usage,
                        "choices": [{"index": 0, "message": msg,
                                     "finish_reason": finish}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                delay = step.get("delay_ms", 0) / 1000
                size = max(1, int(step.get("chunk_chars") or 12))

                def send(delta=None, fin=None, use=None):
                    obj = {"id": cid, "object": "chat.completion.chunk",
                           "created": 0, "model": mock.model,
                           "choices": [] if use else [
                               {"index": 0, "delta": delta or {},
                                "finish_reason": fin}]}
                    if use:
                        obj["usage"] = use
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                send({"role": "assistant", "content": ""})
                if reasoning:
                    for i in range(0, len(reasoning), size):
                        send({"reasoning_content": reasoning[i:i + size]})
                for i in range(0, len(content), size):
                    if delay:
                        time.sleep(delay)
                    send({"content": content[i:i + size]})
                for t in tcs:
                    send({"tool_calls": [t]})
                send(fin=finish)
                if (body.get("stream_options") or {}).get("include_usage"):
                    send(use=usage)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self._srv = ThreadingHTTPServer((host, port), Handler)
        self._srv.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        h, p = self._srv.server_address[:2]
        return f"http://{h}:{p}/v1"

    def start(self):
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    m = MockLLM(a.scenario, port=a.port)
    print(m.base_url, flush=True)
    try:
        m._srv.serve_forever()
    except KeyboardInterrupt:
        pass
