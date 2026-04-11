from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import RAGConfig, EmbeddingsConfig
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".asm": "asm",
    ".s": "asm",
    ".sh": "bash",
    ".bash": "bash",
}

TREE_SITTER_LANG = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "kotlin": "kotlin",
    "c": "c",
    "cpp": "cpp",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "bash": "bash",
}

CHUNK_NODE_TYPES = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "javascript": ["function_declaration", "function_expression", "arrow_function", "class_declaration"],
    "typescript": ["function_declaration", "function_expression", "arrow_function", "class_declaration"],
    "kotlin": ["function_declaration", "class_declaration", "object_declaration"],
    "c": ["function_definition"],
    "cpp": ["function_definition", "class_specifier"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "java": ["method_declaration", "class_declaration", "interface_declaration"],
    "bash": ["function_definition"],
}


def _chunk_id(path: str, start_byte: int) -> str:
    return hashlib.sha256(f"{path}:{start_byte}".encode()).hexdigest()[:16]


from agent._tokens import count_tokens_approx as _count_tokens_approx

_parser_cache: dict = {}


def _get_parser(language: str):
    if language in _parser_cache:
        return _parser_cache[language]
    try:
        from tree_sitter import Language, Parser
        mod_map = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "c": "tree_sitter_c",
            "cpp": "tree_sitter_cpp",
            "rust": "tree_sitter_rust",
            "go": "tree_sitter_go",
            "java": "tree_sitter_java",
            "kotlin": "tree_sitter_kotlin",
            "bash": "tree_sitter_bash",
        }
        mod_name = mod_map.get(language)
        if not mod_name:
            _parser_cache[language] = None
            return None
        import importlib
        mod = importlib.import_module(mod_name)
        lang = Language(mod.language())
        parser = Parser(lang)
        _parser_cache[language] = parser
        return parser
    except Exception:
        _parser_cache[language] = None
        return None


def _extract_name(node, language: str) -> str | None:
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return child.text.decode("utf-8", errors="replace")
    return None


def _parse_with_tree_sitter(content: str, language: str, path: str, cfg: "RAGConfig") -> list[dict]:
    parser = _get_parser(language)
    if parser is None:
        return _fallback_chunks(content, path, language, cfg)

    tree = parser.parse(content.encode("utf-8", errors="replace"))
    root = tree.root_node

    target_types = set(CHUNK_NODE_TYPES.get(language, []))
    chunks = []

    lines = content.splitlines(keepends=True)

    def node_text(node) -> str:
        start = node.start_byte
        end = node.end_byte
        return content[start:end]

    def collect(node, depth=0):
        if node.type in target_types and depth <= 2:
            text = node_text(node)
            token_count = _count_tokens_approx(text)
            if token_count < cfg.chunk_min_tokens:
                return
            name = _extract_name(node, language)
            # For class definitions, emit header + per-method chunks
            if "class" in node.type:
                # Emit class header (up to first method)
                header_lines = []
                for child in node.children:
                    if child.type in target_types:
                        # emit methods separately
                        collect(child, depth + 1)
                    else:
                        header_lines.append(node_text(child))
                header = "\n".join(header_lines[:5])
                if _count_tokens_approx(header) >= cfg.chunk_min_tokens:
                    chunks.append({
                        "id": _chunk_id(path, node.start_byte),
                        "path": path,
                        "language": language,
                        "node_type": node.type + "_header",
                        "name": name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": text[:cfg.chunk_max_tokens * 4],
                    })
                return
            # Trim to max tokens
            if token_count > cfg.chunk_max_tokens:
                text = text[:cfg.chunk_max_tokens * 4]
            chunks.append({
                "id": _chunk_id(path, node.start_byte),
                "path": path,
                "language": language,
                "node_type": node.type,
                "name": name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "content": text,
            })
        else:
            for child in node.children:
                collect(child, depth)

    collect(root)

    # Group remaining top-level statements not covered by chunks
    covered_lines: set[int] = set()
    for c in chunks:
        covered_lines.update(range(c["start_line"], c["end_line"] + 1))

    uncovered: list[str] = []
    uncovered_bytes = 0
    start_line = None
    for i, line in enumerate(lines, 1):
        if i not in covered_lines:
            if start_line is None:
                start_line = i
            uncovered.append(line)
            uncovered_bytes += len(line)
            # flush when we hit chunk_max_tokens
            if uncovered_bytes // 4 >= cfg.chunk_max_tokens:
                if uncovered_bytes // 4 >= cfg.chunk_min_tokens:
                    chunks.append({
                        "id": _chunk_id(path, start_line * 100),
                        "path": path,
                        "language": language,
                        "node_type": "top_level",
                        "name": None,
                        "start_line": start_line,
                        "end_line": i,
                        "content": "".join(uncovered),
                    })
                uncovered = []
                uncovered_bytes = 0
                start_line = None
        else:
            if uncovered:
                if uncovered_bytes // 4 >= cfg.chunk_min_tokens:
                    chunks.append({
                        "id": _chunk_id(path, (start_line or 1) * 100),
                        "path": path,
                        "language": language,
                        "node_type": "top_level",
                        "name": None,
                        "start_line": start_line,
                        "end_line": i - 1,
                        "content": "".join(uncovered),
                    })
                uncovered = []
                uncovered_bytes = 0
                start_line = None

    if uncovered:
        if uncovered_bytes // 4 >= cfg.chunk_min_tokens:
            chunks.append({
                "id": _chunk_id(path, (start_line or 1) * 100),
                "path": path,
                "language": language,
                "node_type": "top_level",
                "name": None,
                "start_line": start_line,
                "end_line": len(lines),
                "content": "".join(uncovered),
            })

    return chunks


def _asm_chunks(content: str, path: str, cfg: "RAGConfig") -> list[dict]:
    """Line-based chunking for assembly files using label boundaries."""
    lines = content.splitlines(keepends=True)
    chunks = []
    current: list[str] = []
    current_bytes = 0
    start_line = 1

    label_re = re.compile(r"^\s*\w+:")

    for i, line in enumerate(lines, 1):
        is_label = label_re.match(line)
        is_blank = not line.strip()

        if (is_label or is_blank) and current:
            if current_bytes // 4 >= cfg.chunk_min_tokens:
                chunks.append({
                    "id": _chunk_id(path, start_line),
                    "path": path,
                    "language": "asm",
                    "node_type": "procedure",
                    "name": None,
                    "start_line": start_line,
                    "end_line": i - 1,
                    "content": "".join(current),
                })
            current = []
            current_bytes = 0
            start_line = i

        current.append(line)
        current_bytes += len(line)

        if current_bytes // 4 >= cfg.chunk_max_tokens:
            chunks.append({
                "id": _chunk_id(path, start_line),
                "path": path,
                "language": "asm",
                "node_type": "procedure",
                "name": None,
                "start_line": start_line,
                "end_line": i,
                "content": "".join(current),
            })
            current = []
            current_bytes = 0
            start_line = i + 1

    if current:
        if current_bytes // 4 >= cfg.chunk_min_tokens:
            chunks.append({
                "id": _chunk_id(path, start_line),
                "path": path,
                "language": "asm",
                "node_type": "procedure",
                "name": None,
                "start_line": start_line,
                "end_line": len(lines),
                "content": "".join(current),
            })

    return chunks


def _fallback_chunks(content: str, path: str, language: str, cfg: "RAGConfig") -> list[dict]:
    lines = content.splitlines(keepends=True)
    chunks = []
    current: list[str] = []
    current_bytes = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        current.append(line)
        current_bytes += len(line)
        if current_bytes // 4 >= cfg.chunk_max_tokens:
            chunks.append({
                "id": _chunk_id(path, start_line),
                "path": path,
                "language": language,
                "node_type": "chunk",
                "name": None,
                "start_line": start_line,
                "end_line": i,
                "content": "".join(current),
            })
            current = []
            current_bytes = 0
            start_line = i + 1

    if current:
        if current_bytes // 4 >= cfg.chunk_min_tokens:
            chunks.append({
                "id": _chunk_id(path, start_line),
                "path": path,
                "language": language,
                "node_type": "chunk",
                "name": None,
                "start_line": start_line,
                "end_line": len(lines),
                "content": "".join(current),
            })

    return chunks


def chunk_file(path: str, cfg: "RAGConfig") -> list[dict]:
    p = Path(path)
    ext = p.suffix.lower()
    language = LANGUAGE_MAP.get(ext)
    if not language:
        return []

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    if language == "asm":
        chunks = _asm_chunks(content, path, cfg)
    elif TREE_SITTER_LANG.get(language):
        chunks = _parse_with_tree_sitter(content, TREE_SITTER_LANG[language], path, cfg)
    else:
        chunks = _fallback_chunks(content, path, language, cfg)

    # Always emit at least one chunk for small files that fell below chunk_min_tokens
    if not chunks:
        lines = content.splitlines()
        chunks = [{
            "id": _chunk_id(path, 0),
            "path": path,
            "language": language,
            "node_type": "file",
            "name": p.name,
            "start_line": 1,
            "end_line": len(lines),
            "content": content,
        }]

    return chunks


def index_directory(
    root: str,
    store: "VectorStore",
    embedder: "Embedder",
    cfg: "RAGConfig",
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
    force: bool = False,
    git_hash: str | None = None,
    progress_cb=None,
) -> dict:
    root_path = Path(root).resolve()
    exclude = exclude or []
    default_exclude = {".git", "__pycache__", "node_modules", "build", "dist", ".agent", ".venv", "venv", ".env"}
    all_exclude = default_exclude | set(exclude)

    allowed_exts: set[str] | None = None
    if languages:
        allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in languages}

    # Load agent rules to respect .agent.ignore during indexing
    from agent.tools.rules import get_rules
    rules = get_rules()

    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in all_exclude]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if allowed_exts and fpath.suffix.lower() not in allowed_exts:
                continue
            if fpath.suffix.lower() not in LANGUAGE_MAP:
                continue
            # Skip files matching .agent.ignore
            rel = str(fpath.relative_to(root_path))
            if rules.ignore.matches(rel):
                continue
            files.append(fpath)

    indexed = 0
    skipped = 0
    total_chunks = 0

    for fpath in files:
        rel = str(fpath.relative_to(root_path))
        mtime = fpath.stat().st_mtime

        if not force:
            stored_mtime = store.get_mtime(rel)
            if stored_mtime is not None and abs(stored_mtime - mtime) < 0.001:
                skipped += 1
                continue

        store.delete_by_path(rel)
        chunks = chunk_file(str(fpath), cfg)
        if not chunks:
            continue

        # Embed in batches of 32
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]
            try:
                embeddings = embedder.embed(texts)
                for chunk, emb in zip(batch, embeddings):
                    chunk["embedding"] = emb
                    chunk["mtime"] = mtime
                    chunk["git_hash"] = git_hash
            except Exception as e:
                for chunk in batch:
                    chunk["mtime"] = mtime
                    chunk["git_hash"] = git_hash

        store.upsert_many(chunks)
        total_chunks += len(chunks)
        indexed += 1

        if progress_cb:
            progress_cb(rel, len(chunks))

    return {"indexed": indexed, "skipped": skipped, "chunks": total_chunks, "files": len(files)}
