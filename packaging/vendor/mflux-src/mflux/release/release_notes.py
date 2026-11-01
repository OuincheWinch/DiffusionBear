import re
import subprocess

import requests

# GitHub PR bodies arrive with \r\n line endings; tolerate the \r on the fence line.
# Both fences are anchored to whole lines — GitHub only renders line-anchored fences,
# so an inline triple-backtick mention in prose must not count as a block.
_FENCE = re.compile(
    r"^```release-note[ \t\r]*\n(.*?)^```[ \t\r]*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

# The squash-merge subject suffix GitHub writes: "title (#123)".
_PR_SUFFIX = re.compile(r"\(#(\d+)\)$")
_REVERT_TITLE = re.compile(r'^Revert "(?P<title>.+)"$')

# Label -> section, first match wins in this order. PRs with no matching label land in
# "Changed"; PRs with no release-note block land in their section flagged for the releaser.
_SECTIONS: list[tuple[str, str]] = [
    ("breaking change", "Breaking"),
    ("bug", "Fixed"),
    ("feature suggestion", "Added"),
    ("enhancement", "Added"),
    ("improvement", "Improved"),
    ("documentation", "Docs"),
    ("chore", "Internal"),
    ("ci", "Internal"),
]
_SECTION_ORDER = ["Breaking", "Added", "Improved", "Fixed", "Docs", "Internal", "Changed"]


class ReleaseNotes:
    @staticmethod
    def extract_release_note(body: str | None) -> str | None:
        # Only the fenced block is trusted: review bots append their own sections to PR
        # bodies, so anything outside the fence is not extractable without guessing.
        if not body:
            return None
        match = _FENCE.search(body)
        if match is None:
            return None
        return match.group(1).strip()

    @staticmethod
    def merged_pr_numbers(previous_tag: str) -> list[int]:
        # The release ships exactly the commits previous_tag..HEAD, so the PR set comes
        # from those commits' squash subjects, not from a time-window search: a PR merged
        # while the approver reviews the draft would fall out of both this release and the
        # next one under `merged:>published_at`, and the publish-time fallback would
        # over-include PRs that are not in the package. Commits without a (#N) suffix
        # (direct pushes) have no PR to harvest and are skipped.
        result = subprocess.run(
            ["git", "log", "--format=%s", f"{previous_tag}..HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        numbers = []
        for subject in result.stdout.splitlines():
            match = _PR_SUFFIX.search(subject.strip())
            if match is not None:
                numbers.append(int(match.group(1)))
        return list(reversed(numbers))  # git log is newest-first; harvest oldest-first

    @staticmethod
    def drop_reverted_pairs(prs: list[dict]) -> list[dict]:
        # A feature merged and reverted within the same release window is a no-op for the
        # user; listing both lines only confuses the notes. Matched by exact title, the
        # way GitHub writes revert PRs.
        titles = {pr["title"]: pr["number"] for pr in prs}
        dropped: set[int] = set()
        for pr in prs:
            match = _REVERT_TITLE.match(pr["title"].strip())
            if match is not None and match.group("title") in titles:
                dropped.add(pr["number"])
                dropped.add(titles[match.group("title")])
        return [pr for pr in prs if pr["number"] not in dropped]

    @staticmethod
    def fetch_pr(github_token: str, github_repo: str, number: int) -> dict:
        response = requests.get(
            f"https://api.github.com/repos/{github_repo}/pulls/{number}",
            headers=ReleaseNotes._headers(github_token),
            timeout=(5, 30),
        )
        if response.status_code != 200:
            raise requests.HTTPError(
                f"GitHub API returned {response.status_code} fetching PR #{number}: {response.text}",
                response=response,
            )
        return response.json()

    @staticmethod
    def latest_release_tag(github_token: str, github_repo: str) -> str:
        response = requests.get(
            f"https://api.github.com/repos/{github_repo}/releases/latest",
            headers=ReleaseNotes._headers(github_token),
            timeout=(5, 30),
        )
        if response.status_code != 200:
            raise requests.HTTPError(
                f"GitHub API returned {response.status_code} resolving the latest release: {response.text}",
                response=response,
            )
        return response.json()["tag_name"]

    @staticmethod
    def render(version: str, prs: list[dict]) -> str:
        sections: dict[str, list[str]] = {name: [] for name in _SECTION_ORDER}
        for pr in sorted(prs, key=lambda p: p["number"]):
            note = ReleaseNotes.extract_release_note(pr.get("body"))
            if note is not None and note.lower() == "none":
                continue
            section = ReleaseNotes._section_for(pr)
            if note:
                lines = [line.strip() for line in note.splitlines() if line.strip()]
                sections[section].append(f"- {' '.join(lines)} (#{pr['number']})")
            else:
                sections[section].append(f"- {pr['title'].strip()} (#{pr['number']}) _[needs edit: no release-note block]_")  # fmt: off

        rendered = [f"## {version}", ""]
        for name in _SECTION_ORDER:
            if not sections[name]:
                continue
            rendered.append(f"### {name}")
            rendered.extend(sections[name])
            rendered.append("")
        return "\n".join(rendered).rstrip() + "\n"

    @staticmethod
    def _section_for(pr: dict) -> str:
        labels = {label["name"].lower() for label in pr.get("labels", [])}
        for label, section in _SECTIONS:
            if label in labels:
                return section
        return "Changed"

    @staticmethod
    def _headers(github_token: str) -> dict:
        return {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }
