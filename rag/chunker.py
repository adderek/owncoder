"""File chunking strategies: tree-sitter, asm, and fallback line-based."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from agent._tokens import count_tokens_approx as _count_tokens_approx

if TYPE_CHECKING:
    from agent.config import RAGConfig

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

_parser_cache: dict = {}


def _chunk_id(path: str, start_byte: int) -> str:
    return hashlib.sha256(f"{path}:{start_byte}".encode()).hexdigest()[:16]


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
        return content[node.start_byte:node.end_byte]

    # Iterative DFS to avoid Python recursion limits on deeply nested ASTs.
    # Stack carries (node, depth, parent_chunk_id) so methods know their class.
    stack = [(root, 0, None)]
    while stack:
        node, depth, parent_chunk_id = stack.pop()
        if node.type in target_types and depth <= 2:
            text = node_text(node)
            token_count = _count_tokens_approx(text)
            if token_count < cfg.chunk_min_tokens:
                continue
            name = _extract_name(node, language)
            chunk_id = _chunk_id(path, node.start_byte)
            if "class" in node.type:
                method_children = []
                # Body nodes (block/class_body) contain methods as grandchildren
                _body_types = {"block", "body", "class_body", "declaration_list"}
                for child in node.children:
                    if child.type in target_types:
                        method_children.append(child)
                    elif child.type in _body_types:
                        for grandchild in child.children:
                            if grandchild.type in target_types:
                                method_children.append(grandchild)
                chunks.append({
                    "id": chunk_id,
                    "path": path,
                    "language": language,
                    "node_type": node.type + "_header",
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "content": text[:cfg.chunk_max_tokens * 4],
                    "parent_chunk_id": parent_chunk_id,
                })
                # Methods belong to this class chunk
                for child in reversed(method_children):
                    stack.append((child, depth + 1, chunk_id))
                continue
            if token_count > cfg.chunk_max_tokens:
                text = text[:cfg.chunk_max_tokens * 4]
            chunks.append({
                "id": chunk_id,
                "path": path,
                "language": language,
                "node_type": node.type,
                "name": name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "content": text,
                "parent_chunk_id": parent_chunk_id,
            })
        else:
            for child in reversed(node.children):
                stack.append((child, depth, parent_chunk_id))

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

    try:
        if language == "asm":
            chunks = _asm_chunks(content, path, cfg)
        elif TREE_SITTER_LANG.get(language):
            chunks = _parse_with_tree_sitter(content, TREE_SITTER_LANG[language], path, cfg)
        else:
            chunks = _fallback_chunks(content, path, language, cfg)
    except RecursionError:
        chunks = _fallback_chunks(content, path, language, cfg)

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
