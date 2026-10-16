#!/usr/bin/env python3
"""Harvest Draw Things name tables out of ModelMapping.swift.

Draw Things maps every diffusers/kohya key to its own internal `[t-N-M]`
identifier in `Libraries/SwiftDiffusion/Sources/Mappings/ModelMapping.swift`
(`drawthingsai/draw-things-community`, revision 060f99d). Those tables are the
authority for reversing a `.ckpt` LoRA, so we harvest them once into JSON
rather than hand-transcribing several thousand rows.

This reads a local checkout. It downloads nothing.

    python tools/harvest_model_mapping.py \
        --source /path/to/ModelMapping.swift \
        --out backend/drawthings_name_map.json

Re-run it whenever the pinned Draw Things revision changes; the generated JSON
is committed so neither the app nor its tests need the Swift source at runtime.
"""

import argparse
import ast
import json
import pathlib
import re
import sys

# The tables a LoRA can reference. Wan, Z-Image, Krea2 and Qwen-Image have no
# literal table -- they are generated at runtime by ModelWeightMapper closures --
# so they are absent here by design and refused at runtime.
WANTED = (
    "UNetXLBase",
    "UNetXLBaseFixed",
    "UNetXLRefiner",
    "UNetXLRefinerFixed",
    "UNet",
    "CLIPTextModel",
    "OpenCLIPTextModel",
    "OpenCLIPTextModelG",
    "OpenCLIPTextModelGTransformers",
)

# The tables come in two syntaxes and the harvester must understand both.
#
# Dictionary literal -- the five UNet tables:
#     "model.diffusion_model.time_embed.0.weight": ["t-0-0"],
#
# Subscript assignment -- the four text-encoder tables, which are closures:
#     textModel["...layers.\(i).layer_norm1.weight"] = ["t-\(i * 8 + 2)-0"]
#
# Reading only one syntax drops the other table's rows *silently*, which is how
# four rows per text table went missing before this was pinned by a test.
TABLE_RE = re.compile(r"public static let (?P<name>\w+): ModelWeightMapping = [\[{]")
LITERAL_RE = re.compile(r'"(?P<key>[^"]*)"\s*:\s*\[(?P<ids>[^\]]*)\]', re.DOTALL)
SUBSCRIPT_RE = re.compile(
    r'(?P<var>\w+)\[\s*"(?P<key>[^"]*)"\s*\]\s*=\s*\[(?P<ids>[^\]]*)\]',
    re.DOTALL,
)
ID_RE = re.compile(r'"(?P<id>t-[^"]+)"')
# Checked *after* the loop variable is substituted in, so a row whose id is
# simply malformed (`t-2-x`) is refused rather than shipped. The grammar is
# the one `drawthings_names.parse_name` enforces.
STRICT_ID_RE = re.compile(r"^t-(?:[^-]+-)?\d+-\d+$")
_PATTERN = "t-<scope>-<layer>-<param>"

# The text-encoder tables are not literal rows: they are built in a Swift loop
# with string interpolation, e.g.
#     for i in 0..<12 {
#       textModel["...encoder.layers.\(i).layer_norm1.weight"] = ["t-\(i * 8 + 2)-0"]
#     }
# so they must be expanded rather than scraped. A row's id list can be longer
# than one entry -- `attn.in_proj_weight` fuses q, k and v -- so the whole list
# is captured and every id expanded. A regex that captured one id and expected
# the closing bracket next would match nothing at all for a fused row and lose
# it silently.
LOOP_RE = re.compile(
    r"for\s+(?P<var>\w+)\s+in\s+0\.\.<(?P<count>\d+)\s*\{"
)
ANY_LOOP_RE = re.compile(r"for\s+[^{]*\{")
# Any assignment whose right-hand side is an array literal. The harvester reads
# exactly two spellings (`LITERAL_RE` and `SUBSCRIPT_RE`); anything else that
# ends in `[...]` would be dropped without a word, which is the same
# silent-row-loss class this tool was rewritten to eliminate.
ANY_ARRAY_ASSIGN_RE = re.compile(r"=\s*\[")
INTERP_RE = re.compile(r"\\\(([^)]*)\)")

def _eval_int(expr, env):
    """Evaluate an integer arithmetic expression with no attribute access.

    Deliberately not `eval`: this reads file contents, so the parser admits
    only integers, the loop variable, and `+ - * // %` on them. Anything else
    -- a call, a subscript, an attribute, a name we did not bind -- raises.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expr!r}: {exc}") from None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise ValueError(f"non-integer constant in {expr!r}")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"unknown name {node.id!r} in {expr!r}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            return left ** right
        raise ValueError(f"disallowed expression {expr!r}")

    return walk(tree)


def _expand(template, var, value):
    """Substitute a Swift loop variable into a string-interpolated template."""

    def repl(match):
        return str(_eval_int(match.group(1), {var: value}))

    return INTERP_RE.sub(repl, template)


def _block_after(text, start):
    """Return the brace-delimited block beginning at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return text[start:]


def _table_spans(text):
    """Yield `(name, body_start, body_end)` for each table, bounded correctly.

    The bound matters: the text-encoder tables reuse the same `t-N-M` ids as
    each other, so a search that runs past a table's end merges one table's
    identifiers into another's and every id becomes ambiguous.
    """
    starts = [(m.group("name"), m.end()) for m in TABLE_RE.finditer(text)]
    for index, (name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        yield name, start, end


def _put(name, rows, key, ids):
    """Add a row, refusing to let a later definition silently overwrite one.

    Two ids for one key means the source table is not what we think it is, and
    the resulting name map would resolve to whichever row happened to be parsed
    last. That is a bug to stop on, not to pick a winner for.
    """
    if key in rows and rows[key] != ids:
        raise SystemExit(
            f"{name}: {key!r} is claimed by both {rows[key]} and {ids}"
        )
    malformed = [i for i in ids if not STRICT_ID_RE.match(i)]
    if malformed:
        raise SystemExit(
            f"{name}: {key!r} has malformed identifier(s) {malformed}; "
            f"expected {_PATTERN}"
        )
    rows[key] = ids


def _loop_bodies(span):
    """Return `(start, end, var, count)` for each loop, innermost first.

    Innermost first because a nested Swift loop reaches the same assignment
    twice; expanding it in the inner loop and skipping the outer keeps the row
    identical instead of raising a spurious conflict.
    """
    bodies = []
    for loop in LOOP_RE.finditer(span):
        start = loop.end() - 1
        body = _block_after(span, start)
        bodies.append((start, start + len(body), loop.group("var"), int(loop.group("count"))))
    bodies.sort(key=lambda b: (-b[0], -b[1]))
    return bodies


def _ids_of(match, name, kind):
    ids = ID_RE.findall(match.group("ids"))
    if not ids:
        raise SystemExit(
            f"{name}: {kind} assignment to {match.group('key')!r} has no "
            f'"t-" identifier in {match.group("ids").strip()!r}; refusing to '
            "harvest a table we cannot read completely"
        )
    return ids


def _interpolated(key, ids):
    return any("\\(" in part for part in (key, *ids))


def _harvest_table(name, span):
    """Return `{diffusers_key: [ids]}` for one table, or fail loudly."""
    rows = {}
    loops = _loop_bodies(span)

    for match in LITERAL_RE.finditer(span):
        _put(name, rows, match.group("key"), _ids_of(match, name, "literal"))

    for match in SUBSCRIPT_RE.finditer(span):
        key, ids = match.group("key"), _ids_of(match, name, "subscript")
        if not _interpolated(key, ids):
            _put(name, rows, key, ids)
            continue
        owner = next((b for b in loops if b[0] <= match.start() < b[1]), None)
        if owner is None:
            raise SystemExit(
                f"{name}: {key!r} is string-interpolated but no loop encloses it; "
                "refusing to harvest a table we cannot read completely"
            )
        _, _, var, count = owner
        for value in range(count):
            _put(
                name,
                rows,
                _expand(key, var, value),
                [_expand(i, var, value) for i in ids],
            )
    return rows


def unrecognised_assignments(text, wanted=WANTED):
    """Return every array assignment in a wanted table we do not read.

    Symmetric to `unrecognised_loops`: the tool already refuses a loop form it
    cannot expand, so it must equally refuse an assignment form it cannot parse
    rather than dropping the row.
    """
    unread = []
    for name, start, end in _table_spans(text):
        if name not in wanted:
            continue
        span = text[start:end]
        consumed = [(m.start(), m.end()) for m in LITERAL_RE.finditer(span)]
        consumed += [(m.start(), m.end()) for m in SUBSCRIPT_RE.finditer(span)]
        # A table's span runs up to the *next* table's `= [`, so the next
        # table's own declaration header lands inside it. Not an assignment.
        consumed += [(m.start(), m.end()) for m in TABLE_RE.finditer(span)]
        for m in ANY_ARRAY_ASSIGN_RE.finditer(span):
            if not any(s <= m.start() < e for s, e in consumed):
                line = span.count("\n", 0, m.start()) + 1
                excerpt = span[max(0, m.start() - 60):m.end() + 40].strip()
                unread.append(f"{name}:{line}: ...{excerpt}...")
    return unread


def unrecognised_loops(text):
    """Return every `for` header that `LOOP_RE` cannot parse.

    Revision 060f99d has four loops, all `for i in 0..<N`. A future revision
    that adds a different form would lose every row inside it without a word,
    so the shape is reported rather than assumed.
    """
    seen = {(m.start(), m.end()) for m in LOOP_RE.finditer(text)}
    return [
        text[m.start():m.end()].strip()
        for m in ANY_LOOP_RE.finditer(text)
        if not any(start <= m.start() < end for start, end in seen)
    ]


def missing_tables(tables, wanted):
    """Return the wanted tables that yielded no rows at all."""
    return [t for t in wanted if t not in tables]


def harvest(text, wanted=WANTED):
    """Return `{table: {diffusers_key: [ids]}}` from ModelMapping.swift source.

    Raises `SystemExit` on anything it cannot read: an assignment with no `t-`
    identifier, an interpolated key no loop encloses, a key with two different
    id lists. A harvest that quietly drops rows is worse than one that refuses,
    because the converter then emits a file that loads and ignores part of the
    LoRA.
    """
    tables = {}
    for name, start, end in _table_spans(text):
        if name not in wanted:
            continue
        rows = _harvest_table(name, text[start:end])
        if rows:
            tables[name] = rows
    unrecognised = unrecognised_loops(text)
    if unrecognised:
        raise SystemExit(
            f"unsupported loop form(s) in the source: {unrecognised}; the rows "
            "inside them would be dropped silently"
        )
    unread = unrecognised_assignments(text, wanted)
    if unread:
        raise SystemExit(
            f"{len(unread)} array assignment(s) in the source are in a form this "
            "tool does not read, so their rows would be dropped silently:\n  "
            + "\n  ".join(unread)
        )
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to ModelMapping.swift")
    parser.add_argument("--out", required=True, help="path to write the JSON to")
    parser.add_argument("--tables", nargs="*", default=list(WANTED))
    args = parser.parse_args(argv)

    text = pathlib.Path(args.source).read_text()
    tables = harvest(text, tuple(args.tables))

    missing = missing_tables(tables, args.tables)
    if missing:
        print(f"error: no rows harvested for {missing}", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables, indent=1, sort_keys=True) + "\n")

    total = sum(len(r) for r in tables.values())
    print(f"wrote {out} -- {len(tables)} tables, {total} rows")
    for name in sorted(tables):
        print(f"  {name:32} {len(tables[name]):>5} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())