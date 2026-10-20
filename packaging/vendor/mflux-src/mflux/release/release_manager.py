import os

import requests

from mflux.release.git_operations import GitOperations
from mflux.release.github_api import GitHubAPI
from mflux.release.pypi_publisher import PyPIPublisher
from mflux.release.release_notes import ReleaseNotes
from mflux.release.release_validator import ReleaseValidator
from mflux.utils.version_util import VersionUtil


class ReleaseManager:
    @staticmethod
    def create_release(
        github_token: str,
        pypi_token: str | None,
        github_repo: str | None = None,
        package_name: str = "mflux",
        trusted_publishing: bool = False,
    ) -> None:
        github_repo = github_repo or os.getenv("GITHUB_REPOSITORY", "mflux-community/mflux")
        # 0. Load version from pyproject.toml
        version = VersionUtil.get_mflux_version()
        tag_name = f"v.{version}"
        print("🚀 Starting MFLUX release process...")
        print(f"📦 Releasing version: {version} (tag: {tag_name}) [from pyproject.toml]")

        # 1. Validate everything is ready for release
        ReleaseValidator.validate_release_ready(version)

        # 2. Check current release state
        git_tag_exists = GitOperations.check_tag_exists(tag_name)
        github_release_exists = GitHubAPI.check_github_release_exists(github_token, github_repo, tag_name)

        # 3. Print release status and exit early if already complete
        ReleaseManager._print_release_status(tag_name, git_tag_exists, github_release_exists)
        if ReleaseManager._is_release_complete(git_tag_exists, github_release_exists):
            return

        # 4. Handle PyPI publishing FIRST (before creating git artifacts)
        if ReleaseManager._should_publish_to_pypi(git_tag_exists, github_release_exists, package_name, version):
            PyPIPublisher.build_and_verify_package()
            if trusted_publishing:
                print("🔐 Trusted publishing enabled — deferring PyPI upload to the workflow publisher step")
            else:
                if not pypi_token:
                    raise ValueError(
                        "PyPI API token is required unless trusted publishing is enabled "
                        "(set MFLUX_TRUSTED_PUBLISHING=true or pass --trusted-publishing)"
                    )
                PyPIPublisher.publish_to_pypi(pypi_token, package_name, version)

        # 5. Create git tag if needed
        if not git_tag_exists:
            GitOperations.create_and_push_tag(tag_name, version)

        # 6. Publish the reviewed draft release (created by the draft-notes job, and
        # possibly edited by whoever approved the pypi deployment). A missing draft
        # falls back to harvesting fresh notes so a manual run still completes.
        if not github_release_exists:
            draft = GitHubAPI.find_release(github_token, github_repo, tag_name)
            if draft is not None and draft.get("draft", False):
                GitHubAPI.publish_draft_release(github_token, github_repo, draft)
            else:
                release_notes = ReleaseManager._harvest_notes(github_token, github_repo, version)
                GitHubAPI.create_github_release(github_token, github_repo, tag_name, version, release_notes)

        print(f"🎉 Release process completed successfully for version {version}!")

    @staticmethod
    def draft_notes(github_token: str, github_repo: str | None = None) -> None:
        github_repo = github_repo or os.getenv("GITHUB_REPOSITORY", "mflux-community/mflux")
        version = VersionUtil.get_mflux_version()
        tag_name = f"v.{version}"
        # Never overwrite: the draft is the approver's working copy (they edit it before the
        # gated job publishes it verbatim), so a re-dispatch must not clobber those edits.
        # Re-harvesting requires deleting the draft first. A published release is the
        # documented re-dispatch no-op, not an error.
        existing = GitHubAPI.find_release(github_token, github_repo, tag_name)
        if existing is not None:
            if existing.get("draft", False):
                print(f"\U0001f4dd Draft for {tag_name} already exists; leaving the approver's copy untouched")
            else:
                print(f"✅ Release {tag_name} is already published; nothing to draft")
            return
        notes = ReleaseManager._harvest_notes(github_token, github_repo, version)
        GitHubAPI.create_draft_release(github_token, github_repo, tag_name, version, notes)

    @staticmethod
    def _harvest_notes(github_token: str, github_repo: str, version: str) -> str:
        # Keyed on commits, not timestamps: the notes must list exactly the PRs whose
        # squash commits are in previous_tag..HEAD, whether this runs at dispatch time
        # (draft-notes job) or as the publish-time fallback.
        previous_tag = ReleaseNotes.latest_release_tag(github_token, github_repo)
        numbers = ReleaseNotes.merged_pr_numbers(previous_tag)
        prs = [ReleaseNotes.fetch_pr(github_token, github_repo, number) for number in numbers]
        prs = ReleaseNotes.drop_reverted_pairs(prs)
        print(f"\U0001f4e5 Harvested {len(prs)} merged PRs in {previous_tag}..HEAD")
        return ReleaseNotes.render(version, prs)

    @staticmethod
    def _is_release_complete(git_tag_exists: bool, github_release_exists: bool) -> bool:
        return git_tag_exists and github_release_exists

    @staticmethod
    def _is_release_partial(git_tag_exists: bool, github_release_exists: bool) -> bool:
        return git_tag_exists or github_release_exists

    @staticmethod
    def _should_publish_to_pypi(
        git_tag_exists: bool,
        github_release_exists: bool,
        package_name: str,
        version: str,
    ) -> bool:
        # Only publish if this is a completely new release
        if ReleaseManager._is_release_partial(git_tag_exists, github_release_exists):
            print("⚠️  Skipping PyPI publishing since this appears to be a re-run")
            return False

        # Check if version already exists on PyPI
        try:
            if PyPIPublisher.version_exists_on_pypi(package_name, version):
                return False
        except (requests.RequestException, ValueError, OSError) as e:
            print(f"❌ Failed to check PyPI version: {e}")
            raise ValueError(f"PyPI version check failed: {e}") from e

        return True

    @staticmethod
    def _print_release_status(tag_name: str, git_tag_exists: bool, github_release_exists: bool) -> None:
        if ReleaseManager._is_release_complete(git_tag_exists, github_release_exists):
            print(f"✅ Release {tag_name} already exists completely")
            print("🔄 This appears to be a re-run of an existing release - nothing to do!")
        elif ReleaseManager._is_release_partial(git_tag_exists, github_release_exists):
            print("⚠️  Partial release state detected:")
            print(f"   Git tag exists: {git_tag_exists}")
            print(f"   GitHub release exists: {github_release_exists}")
            print("   Will complete the missing parts...")
