"""Static smoke guards for the Engine settings panel.

Regression: the auto wired-budget work shipped a component that COMPILED and then threw
`ReferenceError: Can't find variable: s` at runtime, so the whole studio fell over to the
recovery boundary. `npm run build` was clean and every static assertion passed -- JSX does
not resolve identifiers, and a regex test cannot either.

The class of defect is "identifier used in a scope that does not bind it". The only thing
that actually catches it is rendering the component. So this file does two things:

  * a cheap structural check that catches the specific mistake again, and
  * a live render against the running app, skipped when it is not up.

Keep the cheap check even if the live one is skipped in CI: it is the one that fails
loudly in a bare checkout.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "settings" / "EngineSection.jsx"


def identifiers_in_render(code: str) -> set:
    """Rough set of names a JSX expression can legally reference in render scope.

    Deliberately permissive: hooks, globals, locale helpers, props and locals declared
    before the return. The point is to catch a name that appears ONLY in JSX and nowhere
    in scope -- which is exactly what `s` was.
    """
    names = set(re.findall(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", code))
    names |= set(re.findall(r"\(([^()]*)\)\s*=>", code))  # destructured params, roughly
    names |= {
        "t", "status", "wired", "wiredGb", "kreaGb", "mfluxIdle", "sdxlIdle", "qwenIdle",
        "draft", "setDraft", "dirty", "setDirty", "saving", "setSaving", "err", "setErr",
        "refreshTimer", "showStderr", "setShowStderr", "gauge", "TuneInput", "engine",
        "sdxl", "mflux", "qwen", "model_info", "refresh", "saveConfig", "normalizeNum",
        "formatBytes", "fmtSeconds", "setInterval", "clearTimeout", "true", "false", "null",
    }
    for blob in re.findall(r"\{([^}]*)\}", code):
        for part in blob.split(","):
            name = part.strip().split(":")[0].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", name or ""):
                names.add(name)
    return names


class EnginePanelIdentifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = SRC.read_text(encoding="utf-8")

    def test_jsx_only_references_identifiers_that_are_in_scope(self):
        """Every dotted reference inside JSX must start with a name that is bound.

        `s.wired?.x` was fine inside the fetch effect, where `s` is the response, and a
        ReferenceError in the render body, where it is not. Static JSX checking cannot
        tell those apart, so this asserts the cheap thing that would have caught the
        specific regression: a name used in JSX that is never declared anywhere.
        """
        bound = identifiers_in_render(self.code)
        # Only look at the JSX return block, after the last hook.
        start = self.code.rindex("  return (")
        jsx = self.code[start:]
        used = set(re.findall(r"\b([a-z][\w$]*)\.(?:wired|status|metal|mflux|sdxl|qwen)\b", jsx))
        self.assertEqual(
            used - bound,
            set(),
            f"JSX uses unbound name(s): {sorted(used - bound)} -- this compiles and then throws",
        )

    def test_the_render_scope_uses_wired_not_s(self):
        """Spelled out because it is the exact regression, and the generic check above
        would be weakened by someone adding `s` to the permissive allowlist."""
        jsx = self.code[self.code.rindex("  return ("):]
        self.assertNotIn("s.wired?.", jsx, "render scope binds the status as `wired`")
        self.assertRegex(jsx, r"wired\.generic_derived_gb")
        self.assertRegex(jsx, r"wired\.krea_derived_gb")

    def test_no_identifier_in_the_file_is_used_only_in_jsx(self):
        code = self.code
        jsx = code[code.rindex("  return ("):]
        body = code[: code.rindex("  return (")]
        for name in set(re.findall(r"\b([a-z][\w$]{0,12})\.(?:wired|status|metal)\b", jsx)):
            with self.subTest(name=name):
                self.assertRegex(
                    body, rf"\b{re.escape(name)}\s*=", f"`{name}` is used in JSX but never assigned"
                )


class LiveRenderTests(unittest.TestCase):
    """Renders the real panel against the running app. Skipped when it is not up."""

    def test_the_engine_panel_renders_without_the_recovery_boundary(self):
        import json
        import shutil
        import subprocess
        import urllib.request

        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        try:
            urllib.request.urlopen("http://127.0.0.1:8001/api/version", timeout=3)
        except Exception:
            self.skipTest("app is not running")

        # playwright is not a dependency of this repo, so find a directory it resolves
        # from rather than assuming one. Skipping is correct here: the static checks above
        # are what must pass in a bare checkout, and a live render is a bonus.
        probe_cwd = None
        for cand in (Path(__file__).parent, Path("/tmp"), Path.home()):
            if not cand.is_dir():
                continue
            probe = cand / "_resolve_pw.mjs"
            probe.write_text('import("playwright").then(()=>process.stdout.write("ok")).catch(()=>process.exit(1))')
            try:
                r = subprocess.run([node, str(probe)], capture_output=True, text=True, timeout=90, cwd=str(cand))
            except Exception:
                r = None
            probe.unlink(missing_ok=True)
            if r and r.returncode == 0 and "ok" in r.stdout:
                probe_cwd = cand
                break
        if probe_cwd is None:
            self.skipTest("playwright is not installed in any candidate directory")

        # The probe must be written INTO probe_cwd, not next to this file: node resolves a
        # bare specifier by walking up from the IMPORTING FILE's directory, so a probe
        # sitting in backend/ can never see a playwright installed in /tmp/node_modules
        # no matter what cwd the process is launched with.
        probe = probe_cwd / "_engine_panel_probe.mjs"
        probe.write_text(
            """
import { chromium } from "playwright";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
await p.goto("http://127.0.0.1:8001/", { waitUntil: "load" });
await p.waitForTimeout(2500);
await p.click("text=Parameters").catch(() => {});
await p.waitForTimeout(2500);
const crashed = await p.evaluate(() => !!document.querySelector(".fatal-recovery, [data-fatal-recovery]"));
const marker = await p.evaluate(() => document.body.innerText.includes("Interface Recovery"));
const auto = await p.evaluate(() => {
  const el = [...document.querySelectorAll(".engine-tune-auto")].map((x) => x.textContent.trim());
  return el;
});
console.log(JSON.stringify({ crashed, marker, auto }));
await b.close();
"""
        )
        try:
            out = subprocess.run(
                ["node", str(probe)], capture_output=True, text=True, timeout=240, cwd=str(probe_cwd)
            )
        finally:
            probe.unlink(missing_ok=True)
        self.assertEqual(out.returncode, 0, f"probe failed: {out.stderr[-400:]}")
        data = json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][-1])
        self.assertFalse(data["marker"], "the studio fell over to the recovery boundary")
        self.assertFalse(data["crashed"], "fatal recovery boundary rendered")
        self.assertTrue(
            data["auto"], "expected two Auto badges showing the derived GB"
        )
