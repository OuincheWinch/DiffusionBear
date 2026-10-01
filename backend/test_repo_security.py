"""Supply-chain / tamper guard for the repository.

Fails the build when a tracked file contains code patterns that have no place in
this project, when outbound network hosts fall outside an explicit allowlist, or
when credentials are committed. Run automatically in CI.
"""

import json
import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).name

CODE_SUFFIXES = {".py", ".js", ".jsx", ".mjs", ".cjs", ".sh", ".bash", ".zsh", ".html", ".css"}
# Executable or CI-controlled content: these are audited for behaviour.
CODE_AND_CONFIG = CODE_SUFFIXES | {".yml", ".yaml", ".toml", ".cfg", ".ini"}

# Offences that are reviewed individually and accepted, listed as exact
# "<file>:<line> <label>" strings so the allowance disappears the moment the code
# moves -- a stale entry is a failing test, not dead config. There is deliberately
# no pattern-level or file-level exemption: those silently disable a check for
# every future line added to that file.
_ALLOWED_DANGEROUS = {
    # fill.py decodes a base64 PNG mask sent by the browser. base64 is an
    # encoding, not a serialisation format: it cannot execute anything, and the
    # result is handed straight to PIL's image loader rather than to pickle,
    # marshal, ctypes or eval. It is size-capped before decode and
    # dimension-checked after, because a mask is untrusted input either way.
    "backend/fill.py:190 base64 decode",
}
# Prose and lockfiles are scanned for secrets only; documentation may link anywhere.
TEXT_SUFFIXES = CODE_AND_CONFIG | {".json", ".md", ".txt"}

# Hosts this project is allowed to talk to.
ALLOWED_HOSTS = {
    "127.0.0.1",
    "localhost",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    # Civitai and its official mirrors are interchangeable download sources.
    "civitai.com",
    "www.civitai.com",
    "civitai.red",
    "civitai.green",
    "www.ouinche.com",
    "ouinche.com",
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
}

DANGEROUS_PATTERNS = {
    # Lookbehind rejects attribute calls: mx.eval(...) is an MLX tensor op, not eval().
    "dynamic eval": r"(?<![.\w])eval\s*\(",
    "dynamic exec": r"(?<![.\w])exec\s*\(",
    # Only flagged when the command is not a plain literal, i.e. shell interpolation.
    "os.system with interpolation": r"\bos\.system\(\s*(?![\"'])",
    "os.popen with interpolation": r"\bos\.popen\(\s*(?![\"'])",
    "subprocess shell=True": r"shell\s*=\s*True",
    "pickle deserialization": r"\bpickle\.loads?\s*\(",
    "marshal deserialization": r"\bmarshal\.loads\s*\(",
    "dunder import": r"__import__\s*\(",
    "ctypes native call": r"\bctypes\.(CDLL|windll|cdll)\b",
    "codecs decode obfuscation": r"\bcodecs\.decode\s*\(",
    "base64 decode": r"\bbase64\.(b64decode|decodebytes|std_b64decode)\s*\(",
    "zlib decompress bomb": r"\bzlib\.decompress\s*\(",
    "chr() string building": r"\bchr\s*\(\s*\d+\s*\)\s*\+\s*chr\s*\(",
    "browser exfiltration": r"\b(navigator\.sendBeacon|new\s+WebSocket|document\.cookie)\b",
    "clipboard read": r"navigator\.clipboard\.read",
}

# Long opaque literals are how payloads hide in source files.
BLOB_LITERAL = re.compile(r"""['"][A-Za-z0-9+/]{140,}={0,2}['"]""")

SECRET_PATTERNS = {
    "huggingface token": r"\bhf_[A-Za-z0-9]{20,}",
    "github token": r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    "aws access key": r"\bAKIA[0-9A-Z]{16}\b",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "slack webhook": r"https://hooks\.slack\.com/services/",
    "discord webhook": r"https://discord(app)?\.com/api/webhooks/",
    "telegram bot": r"api\.telegram\.org/bot",
}

NPM_ALLOWED_DEPS = {"react", "react-dom"}
NPM_ALLOWED_DEV_DEPS = {"@vitejs/plugin-react", "oxlint", "vite"}
NPM_FORBIDDEN_SCRIPTS = {"preinstall", "install", "postinstall", "prepare", "prepublish", "prepack"}


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _tracked_files():
    try:
        raw = _git("ls-files", "-z")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise unittest.SkipTest("not a git checkout; nothing to audit")
    return [REPO_ROOT / name for name in raw.split("\0") if name]


def _read(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


class MaliciousCodeTests(unittest.TestCase):
    def setUp(self):
        self.files = [path for path in _tracked_files() if path.name != SELF and path.is_file()]
        # packaging/vendor/ is a vendored copy of an upstream package (mflux, with the
        # Qwen-Image 2.1 port), not our code. Its outbound hosts are upstream's and it
        # is fetched from PyPI in every other context, so host-scanning it would only
        # flag upstream. It is covered by a provenance test instead --
        # test_vendored_mflux_is_unmodified -- which pins a hash of the whole tree, so
        # tampering is still caught and is caught more strictly than a host list.
        self.own_files = [p for p in self.files if "packaging/vendor/" not in p.as_posix()]

    def test_repository_has_files_to_audit(self):
        self.assertGreater(len(self.files), 20)

    def test_no_dangerous_code_patterns(self):
        offences = []
        for path in self.own_files:
            if path.suffix not in CODE_AND_CONFIG:
                continue
            text = _read(path)
            if text is None:
                continue
            for label, pattern in DANGEROUS_PATTERNS.items():
                for match in re.finditer(pattern, text):
                    line = text.count("\n", 0, match.start()) + 1
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    offence = f"{rel}:{line} {label}"
                    if offence in _ALLOWED_DANGEROUS:
                        continue
                    offences.append(offence)
        self.assertEqual(offences, [], "dangerous patterns found:\n" + "\n".join(offences))

    def test_no_obfuscated_blob_literals(self):
        offences = []
        for path in self.own_files:
            if path.suffix not in CODE_SUFFIXES:
                continue
            text = _read(path)
            if text and BLOB_LITERAL.search(text):
                offences.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offences, [], "opaque literals found:\n" + "\n".join(offences))

    def test_outbound_hosts_are_allowlisted(self):
        offences = []
        for path in self.own_files:
            if path.suffix not in CODE_AND_CONFIG:
                continue
            text = _read(path)
            if not text:
                continue
            for host in re.findall(r"https?://([A-Za-z0-9._-]+)", text):
                if host.lower() not in ALLOWED_HOSTS:
                    offences.append(f"{path.relative_to(REPO_ROOT)} -> {host}")
        self.assertEqual(sorted(set(offences)), [], "unexpected hosts:\n" + "\n".join(sorted(set(offences))))

    def test_vendored_mflux_is_unmodified(self):
        """The vendored fork is exempt from the host scan, so pin its exact contents.

        packaging/vendor/mflux-src is upstream mflux 0.20.0 plus the Qwen-Image 2.1
        port. It is excluded from the outbound-host and code-pattern scans as
        third-party code, which makes a hash of the whole tree the guard that
        matters: any edit, added file, or removed file changes the digest.
        """
        import hashlib

        root = REPO_ROOT / "packaging" / "vendor" / "mflux-src"
        self.assertTrue(root.is_dir(), "vendored mflux is missing; the bundle build needs it")
        digest = hashlib.sha256()
        count = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode())
            digest.update(b"\0")
            count += 1
        self.assertEqual(count, 816, f"vendored file count changed (was 816, now {count})")
        self.assertEqual(
            digest.hexdigest(),
            "732b9193b0c965ba70493083a853dc30c5e3c66827933b934ae69d1b487def5c",
            "vendored mflux has been modified; if that is intended, re-pin this digest",
        )

    def test_no_committed_credentials(self):
        offences = []
        for path in self.own_files:
            if path.suffix not in TEXT_SUFFIXES:
                continue
            text = _read(path)
            if not text:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if re.search(pattern, text):
                    offences.append(f"{path.relative_to(REPO_ROOT)} {label}")
        self.assertEqual(offences, [], "credential-like content:\n" + "\n".join(offences))

    def test_npm_manifest_has_no_lifecycle_hooks(self):
        manifest = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        hooks = NPM_FORBIDDEN_SCRIPTS & set(manifest.get("scripts") or {})
        self.assertEqual(hooks, set(), f"npm lifecycle hooks present: {hooks}")
        self.assertEqual(set(manifest.get("dependencies") or {}), NPM_ALLOWED_DEPS)
        self.assertEqual(set(manifest.get("devDependencies") or {}), NPM_ALLOWED_DEV_DEPS)

    def test_npm_lockfile_is_committed(self):
        self.assertTrue((REPO_ROOT / "frontend" / "package-lock.json").is_file())

    def test_python_dependencies_are_pinned_or_pypi(self):
        offences = []
        for name in ("requirements.txt", "requirements-sdxl.txt"):
            path = REPO_ROOT / "backend" / name
            for line in _read(path).splitlines():
                line = line.strip()
                if line.startswith(("#", "-")) or not line:
                    continue
                if line.startswith(("git+", "http://", "https://", "-e ", "--index-url")):
                    if not re.search(r"@[0-9a-f]{40}", line):
                        offences.append(f"{name}: {line}")
        self.assertEqual(offences, [], "unpinned remote dependency:\n" + "\n".join(offences))

    def test_downloads_verify_checksums(self):
        for name in ("hf_service.py", "civitai_service.py"):
            text = _read(REPO_ROOT / "backend" / name)
            self.assertIn("sha256", text, f"{name} must verify SHA-256 of downloads")

    def test_gitignore_protects_secrets_and_data(self):
        ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("token", "backend/data/generated/", "backend/data/uploads/", "venv/", "node_modules/"):
            self.assertIn(pattern, ignored, f".gitignore must cover {pattern}")

    def test_no_active_git_hooks_or_attributes(self):
        hooks = REPO_ROOT / ".git" / "hooks"
        if hooks.is_dir():
            active = [p.name for p in hooks.iterdir() if p.is_file() and not p.name.endswith(".sample")]
            self.assertEqual(active, [], f"active git hooks: {active}")
        self.assertFalse((REPO_ROOT / ".gitattributes").exists(), ".gitattributes could rewrite content on checkout")

    def test_workflows_have_no_untrusted_triggers(self):
        workflows = REPO_ROOT / ".github" / "workflows"
        for path in workflows.glob("*.y*ml"):
            text = _read(path)
            self.assertNotIn("pull_request_target", text, f"{path.name} uses pull_request_target")
            self.assertNotIn("secrets.", text, f"{path.name} touches repository secrets")

    def test_tracked_executables_are_shell_scripts_only(self):
        try:
            listing = _git("ls-files", "-s")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("git unavailable")
        offenders = []
        for line in listing.splitlines():
            meta, _, name = line.partition("\t")
            mode = meta.split()[0]
            if mode == "100755" and not name.endswith((".sh", ".bash", ".zsh")):
                offenders.append(name)
        self.assertEqual(offenders, [], "unexpected executable files:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
