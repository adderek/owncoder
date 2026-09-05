"""Images in the prompt: file on disk → OpenAI multimodal content block.

The turn engine keeps `content` a plain string everywhere — compaction, token
counting, transcript rendering and every session store are written against
that, and making them list-aware would touch the whole codebase for one
feature. So an image stays a *marker* in the text (`[image: path]`) for the
entire internal life of a message, and is expanded into real content blocks
only at the API boundary (turn_setup.normalize_api_messages), on the wire copy.

Consequences, all of them wanted:
  - history stays greppable and diffable; a session file has no megabyte blobs,
  - the marker is readable to a text-only model too (it is a path its file
    tools can open), so an image never breaks a non-vision model,
  - only the last few turns' images are actually re-sent (see keep_last_turns),
    which is what keeps a long session from re-uploading every screenshot on
    every request.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Only formats every OpenAI-compatible vision backend accepts. Anything else
# (heic, tiff, svg) stays a text path — a wrong-format data URI is a 400 with
# a confusing message, and a file path always degrades gracefully.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

MARKER_RE = re.compile(r"\[image:\s*([^\]\n]+?)\s*\]")

# Model-id hints for vision, used when nothing is declared in config. Same
# shape as model_probe._hints_thinking: cheap, no network, overridable.
_VISION_HINTS = (
    "-vl", "vl-", "vision", "llava", "moondream", "minicpm-v", "internvl",
    "pixtral", "molmo", "cogvlm", "bakllava", "qwen-vl", "gemma-3", "gemma3",
    "gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4-mini", "claude-3", "claude-4",
    "claude-opus", "claude-sonnet", "claude-haiku", "gemini-", "llama-3.2-11b",
    "llama-3.2-90b", "llama-4", "mistral-small-3", "step-1v", "glm-4v",
)


def hints_vision(model_id: str) -> bool:
    """True when the model id itself says it is multimodal."""
    lowered = (model_id or "").lower()
    return any(kw in lowered for kw in _VISION_HINTS)


def _vision_cfg(config):
    return getattr(config, "vision", None)


def supports_images(config) -> bool:
    """Can the *currently active* endpoint be sent image blocks?

    Order: explicit [agent].vision / [models.<x>].vision → model-id hint.
    """
    cfg = _vision_cfg(config)
    if cfg is not None and not cfg.enabled:
        return False
    mode = str(getattr(getattr(config, "llm", None), "vision", "auto") or "auto").lower()
    if mode in ("on", "true", "yes", "1"):
        return True
    if mode in ("off", "false", "no", "0"):
        return False
    return hints_vision(getattr(getattr(config, "llm", None), "model", ""))


def _workdir(config) -> Path:
    wd = getattr(getattr(config, "tools", None), "working_dir", "") or os.getcwd()
    return Path(wd).resolve()


def resolve_image(raw_path: str, config) -> tuple[Path | None, str]:
    """Resolve a marker path to a readable image inside the working dir.

    Returns (path, reason-if-rejected). The containment check is the point:
    an image block ships the file's bytes to whatever endpoint is active,
    possibly a remote one, so a marker must never be able to name
    ~/.ssh/id_rsa.png or /etc/anything.
    """
    root = _workdir(config)
    p = Path(os.path.expanduser(raw_path.strip()))
    if not p.is_absolute():
        p = root / p
    try:
        p = p.resolve()
    except OSError as exc:
        return None, f"unreadable path ({exc})"
    try:
        p.relative_to(root)
    except ValueError:
        return None, "outside the working directory"
    if not p.is_file():
        return None, "no such file"
    if p.suffix.lower() not in IMAGE_EXTS:
        return None, f"unsupported image type {p.suffix or '(none)'}"
    return p, ""


def _shrink(raw: bytes, path: Path, max_side: int) -> tuple[bytes, str]:
    """Downscale to *max_side* on the long edge when Pillow is available.

    Pillow is optional (owncoder does not depend on torch/PIL), so without it
    the bytes go as they are and the size cap below is the only guard.
    """
    if max_side <= 0:
        return raw, ""
    try:
        import io

        from PIL import Image
    except Exception:
        return raw, ""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size) <= max_side:
                return raw, ""
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        logger.debug("vision: downscale failed for %s", path, exc_info=True)
        return raw, ""


def encode_image(path: Path, config) -> tuple[str, str]:
    """Read an image and return (data-uri, reason-if-rejected)."""
    cfg = _vision_cfg(config)
    max_side = getattr(cfg, "max_side", 1568) if cfg else 1568
    max_bytes = getattr(cfg, "max_bytes", 5_000_000) if cfg else 5_000_000
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return "", f"read failed ({exc})"
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    raw, new_mime = _shrink(raw, path, max_side)
    if new_mime:
        mime = new_mime
    if max_bytes and len(raw) > max_bytes:
        return "", (f"{len(raw) // 1024}KB exceeds the {max_bytes // 1024}KB cap "
                    "(install Pillow to auto-downscale)")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", ""


def image_markers(text: str) -> list[str]:
    """Paths named by `[image: ...]` markers in *text*, in order."""
    return [m.group(1) for m in MARKER_RE.finditer(text or "")]


# Rough per-image prompt cost. Backends differ wildly (llama.cpp ~ hundreds per
# tile, OpenAI ~765 for a 1024² "high" image), and there is no way to know
# before sending — this is a deliberate over-estimate so the context budget
# errs towards compacting early rather than a 400 "context exceeded".
TOKENS_PER_IMAGE = 1200


def estimated_image_tokens(messages: list[dict], config) -> int:
    """What the images in *messages* will cost on the wire, approximately.

    Zero when the model has no vision — then the markers stay plain text and
    the ordinary text count is already right.
    """
    if not supports_images(config):
        return 0
    cfg = _vision_cfg(config)
    keep_last = getattr(cfg, "keep_last_turns", 2) if cfg else 2
    max_images = getattr(cfg, "max_images", 4) if cfg else 4
    counts = [len(image_markers(m.get("content", "")))
              for m in messages
              if m.get("role") == "user" and isinstance(m.get("content"), str)]
    counts = [c for c in counts if c]
    if keep_last > 0:
        counts = counts[-keep_last:]
    n = sum(counts)
    if max_images > 0:
        n = min(n, max_images)
    return n * TOKENS_PER_IMAGE


def expand_image_markers(api_messages: list[dict], config) -> list[dict]:
    """Turn `[image: path]` markers into image_url blocks on the wire copy.

    No-op (returns the input) when the active model has no vision, when the
    feature is off, or when nothing is marked — so this is safe to call on
    every request.
    """
    if not api_messages or not supports_images(config):
        return api_messages
    cfg = _vision_cfg(config)
    keep_last = getattr(cfg, "keep_last_turns", 2) if cfg else 2
    max_images = getattr(cfg, "max_images", 4) if cfg else 4
    detail = getattr(cfg, "detail", "auto") if cfg else "auto"

    # Indices of user messages carrying markers, newest first: older turns'
    # images are dropped back to their text marker so a long session does not
    # re-send every screenshot it ever saw on every request.
    marked = [i for i, m in enumerate(api_messages)
              if m.get("role") == "user" and isinstance(m.get("content"), str)
              and MARKER_RE.search(m["content"])]
    if not marked:
        return api_messages
    live = set(marked[-keep_last:] if keep_last > 0 else marked)

    out = list(api_messages)
    budget = max_images if max_images > 0 else len(marked) * 16
    for idx in sorted(live, reverse=True):     # newest turn gets the budget first
        text = out[idx]["content"]
        blocks: list[dict] = []
        notes: list[str] = []
        for raw_path in image_markers(text):
            if budget <= 0:
                notes.append(f"[image: {raw_path} not sent: per-request image limit reached]")
                continue
            path, why = resolve_image(raw_path, config)
            if path is None:
                notes.append(f"[image: {raw_path} not sent: {why}]")
                continue
            uri, why = encode_image(path, config)
            if not uri:
                notes.append(f"[image: {raw_path} not sent: {why}]")
                continue
            block = {"type": "image_url", "image_url": {"url": uri}}
            if detail and detail != "auto":
                block["image_url"]["detail"] = detail
            blocks.append(block)
            budget -= 1
        if not blocks and not notes:
            continue
        text_part = text if not notes else text + "\n" + "\n".join(notes)
        out[idx] = {**out[idx], "content": [{"type": "text", "text": text_part}] + blocks}
    return out
