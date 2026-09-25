#!/usr/bin/env python3
"""Check DenseCore source ownership without compiling or third-party scanning.

This is a lexical boundary guard, not a substitute for C++ compilation/linking.
Public CpuBackend compatibility overload declarations can mention BatchSpec;
production backend implementations may not access runtime state.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys

SUFFIXES = {".h", ".hpp", ".c", ".cpp", ".cc", ".mm", ".inl"}
INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)
COMMENTS = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)
STRINGS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
FORBIDDEN_SYMBOL = re.compile(
    r'\b(?:InferenceWorkContext|BatchSpec|InferenceConfig|'
    r'GetCurrentWorkContext|GetCurrentBatch|GetCurrentExecutionPhase|'
    r'SetCurrentWorkContext|SetCurrentBatch|SetCurrentExecutionPhase|GetInferenceWorkContextProfile|g_shared_batch|ResolveInferenceConfig)\b|'
    r'\b(?:densecore::)?llm\s*::'
)
# Diagnostic parsing is a pure utility. It does not carry engine state or types.
UTILITY_INCLUDES = {"runtime/runtime_env.h"}


def without_comments(text: str) -> str:
    return COMMENTS.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def production_text(text: str) -> str:
    """Omit explicit test-only branches; inspect unknown platform branches."""
    result: list[str] = []
    stack: list[bool | None] = []
    for line in text.splitlines(keepends=True):
        directive = re.match(r'\s*#\s*(ifdef|ifndef|if|else|elif|endif)\b(.*)', line)
        if directive:
            op, condition = directive.groups()
            if op in {"ifdef", "ifndef", "if"}:
                condition = condition.strip()
                test_only = condition == "DENSECORE_TEST_BUILD" if op != "if" else bool(
                    re.fullmatch(r'defined\s*\(?\s*DENSECORE_TEST_BUILD\s*\)?', condition)
                )
                stack.append(op == "ifndef" if test_only else None)
            elif op == "else" and stack and stack[-1] is not None:
                stack[-1] = not stack[-1]
            elif op == "elif" and stack:
                stack[-1] = None
            elif op == "endif" and stack:
                stack.pop()
        result.append(line if False not in stack else "\n")
    return "".join(result)


def source_files(core: Path):
    for directory in (core / "src", core / "include"):
        for path in sorted(directory.rglob("*")):
            if path.suffix in SUFFIXES:
                yield path


def resolve_include(core: Path, owner: Path, name: str) -> Path | None:
    for candidate in (owner.parent / name, core / "src" / name, core / "include" / name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def check_sources(core: Path) -> list[str]:
    errors: list[str] = []
    implementation_owners: dict[Path, list[Path]] = collections.defaultdict(list)
    for path in source_files(core):
        rel = path.relative_to(core).as_posix()
        text = without_comments(path.read_text())
        backend = rel.startswith(("src/backend/", "include/densecore/backend/"))
        if backend:
            live = production_text(text)
            for match in INCLUDE.finditer(live):
                include = match.group(1)
                if re.match(r'(?:densecore/)?(?:runtime|llm)/', include) and include not in UTILITY_INCLUDES:
                    errors.append(f"{rel}: backend includes upper-layer header {include}")
            # Legacy public C++ overload declarations retain source compatibility;
            # their implementations live above the backend layer.
            for match in FORBIDDEN_SYMBOL.finditer(STRINGS.sub("", live)):
                if rel.startswith("include/") and match.group(0) == "BatchSpec":
                    continue
                errors.append(f"{rel}: backend uses runtime symbol {match.group(0)}")
        for match in INCLUDE.finditer(text):
            include = match.group(1)
            if not include.endswith(".inl"):
                continue
            target = resolve_include(core, path, include)
            if target is None:
                errors.append(f"{rel}: unresolved implementation include {include}")
                continue
            if path.suffix in {".h", ".hpp"}:
                errors.append(f"{rel}: declaration header includes implementation {include}")
            implementation_owners[target].append(path)
    for implementation, owners in implementation_owners.items():
        if len(owners) != 1:
            names = ", ".join(p.relative_to(core).as_posix() for p in owners)
            errors.append(f"{implementation.relative_to(core)}: implementation has multiple owners: {names}")
    return errors


def expected_module(source: str) -> str | None:
    if source in {"src/runtime/graph_executor.cpp", "src/hal/operation_graph.cpp", "src/hal/op_registry.cpp"}:
        return "generic"
    parts = source.split("/")
    if len(parts) < 3 or parts[0] != "src":
        return None
    families = {
        "api": "api", "backend": "backend_kernels", "kernels": "backend_kernels", "simd": "backend_kernels",
        "models": "models_llm", "llm": "models_llm", "moe": "models_llm", "sampler": "models_llm",
        "runtime": "runtime", "hal": "common", "common": "common", "graph": "generic",
        "graph_builders": "generic", "quantization": "tools", "pruning": "tools", "tools": "tools",
    }
    return families.get(parts[1])


def check_modules(core: Path, build: Path) -> list[str]:
    errors: list[str] = []
    manifests = sorted(build.glob("densecore-module-map-*.txt"))
    if not (build / "densecore-module-map-production.txt").exists():
        return [f"{build}: missing production module map; configure this source tree first"]
    sources_by_variant: dict[str, set[str]] = {}
    mapped: dict[tuple[Path, str], str] = {}
    for manifest in manifests:
        variant = manifest.stem.removeprefix("densecore-module-map-")
        seen: set[str] = set()
        for line in manifest.read_text().splitlines():
            fields = line.split("|")
            if len(fields) != 3:
                errors.append(f"{manifest.name}: malformed source mapping {line!r}")
                continue
            source, module, target = fields
            if source in seen:
                errors.append(f"{manifest.name}: duplicate source {source}")
            seen.add(source)
            if not (core / source).is_file():
                errors.append(f"{manifest.name}: missing source {source}")
            if expected_module(source) != module or target != f"densecore_{module}_{variant}":
                errors.append(f"{manifest.name}: wrong owner for {source}: {module}/{target}")
            mapped[((core / source).resolve(), variant)] = target
        sources_by_variant[variant] = seen
    for variant, sources in sources_by_variant.items():
        if sources != sources_by_variant.get("production"):
            errors.append(f"{variant}: source inventory differs from production")
    database = build / "compile_commands.json"
    if database.exists():
        found: collections.Counter[tuple[Path, str]] = collections.Counter()
        for entry in json.loads(database.read_text()):
            command = entry.get("command", " ".join(entry.get("arguments", [])))
            match = re.search(r'CMakeFiles/(densecore_\w+_(production|test|probe))\.dir', command)
            if not match:
                continue
            key = (Path(entry["file"]).resolve(), match.group(2))
            found[key] += 1
            if mapped.get(key) != match.group(1):
                errors.append(f"compile_commands: unexpected source/target {entry['file']} / {match.group(1)}")
        for key, target in mapped.items():
            if found[key] != 1:
                errors.append(f"compile_commands: expected one compilation of {key[0]} in {target}, got {found[key]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    core = args.core_dir.resolve()
    errors = check_sources(core)
    if args.build_dir:
        errors.extend(check_modules(core, args.build_dir.resolve()))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Architecture boundaries: PASS (backend dependencies, single implementation owners" +
          (", configured module inventory" if args.build_dir else "") + ")")
    if args.build_dir and not (args.build_dir / "compile_commands.json").exists():
        print("Compile-command ownership not checked: compilation database is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
