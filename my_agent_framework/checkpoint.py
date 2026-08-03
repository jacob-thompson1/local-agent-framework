"""Workspace checkpointing and rollback.

Snapshots a declared workspace directory after each applied WRITE/DESTRUCTIVE
tool call so an agent mistake is recoverable. Two backends, probed at first
use (never at import):

* **git** (preferred): a *shadow* repository whose ``GIT_DIR`` lives under the
  platform data dir -- the workspace itself gains no ``.git`` and any real
  repository history in it is untouched. Restores are ``git restore`` from
  the checkpoint commit.
* **copy** (fallback when the git binary is absent): full directory snapshots
  under the same data dir. Restore copies files back over the workspace.
  Limitation (documented, logged): files *created after* the checkpoint are
  not deleted on restore -- git mode handles those correctly.

Three restore modes, matching Claude Code's mental model:

* ``files`` -- roll the workspace back, keep the conversation
* ``task``  -- roll the conversation back (memory truncation), keep files
* ``both``  -- full rewind

Each checkpoint records the ConversationMemory turn count at creation time,
which is what ``task`` restore truncates to.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("my_agent_framework.checkpoint")

DEFAULT_IGNORES = (".git", "__pycache__", ".venv", "node_modules", ".pytest_cache")


def default_checkpoint_root() -> Path:
    from platformdirs import user_data_dir  # lazy; tiny, offline

    return Path(user_data_dir("my-agent-framework", appauthor=False)) / "checkpoints"


@dataclass
class Checkpoint:
    checkpoint_id: str
    label: str
    created_ts: float
    memory_turns: int
    backend: str  # "git" | "copy"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class WorkspaceCheckpointer:
    """Checkpoint/restore for one workspace directory.

    Parameters
    ----------
    workspace_root:
        The directory under agent control. Must exist.
    shadow_root:
        Where checkpoint data lives (default: platform data dir). Never
        inside the workspace.
    ignores:
        Directory/file names excluded from snapshots (both backends).
    """

    def __init__(
        self,
        workspace_root: str,
        shadow_root: Optional[str] = None,
        ignores: tuple = DEFAULT_IGNORES,
    ) -> None:
        self.workspace = Path(workspace_root).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace '{self.workspace}' is not a directory.")
        base = Path(shadow_root).resolve() if shadow_root else default_checkpoint_root()
        # one shadow per workspace, keyed by a path digest
        import hashlib
        key = hashlib.sha256(str(self.workspace).encode()).hexdigest()[:16]
        self.shadow = base / key
        self.ignores = tuple(ignores)
        self.checkpoints: list[Checkpoint] = []
        self._backend: Optional[str] = None  # probed on first checkpoint

    # -- backend probing ---------------------------------------------------

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._backend = "git" if shutil.which("git") else "copy"
            if self._backend == "copy":
                logger.warning(
                    "git binary not found; using copy-based snapshots. "
                    "Restores will not delete files created after a "
                    "checkpoint. Install git for exact rollback."
                )
        return self._backend

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        import os
        env = {
            **os.environ,
            "GIT_DIR": str(self.shadow / "repo.git"),
            "GIT_WORK_TREE": str(self.workspace),
            # deterministic, no user config required
            "GIT_AUTHOR_NAME": "my-agent-framework",
            "GIT_AUTHOR_EMAIL": "agent@localhost",
            "GIT_COMMITTER_NAME": "my-agent-framework",
            "GIT_COMMITTER_EMAIL": "agent@localhost",
            "HOME": str(self.shadow),          # isolate from user gitconfig
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        if args and args[0] == "init":
            env.pop("GIT_WORK_TREE", None)  # --bare rejects a work tree
        return subprocess.run(
            ["git", *args], env=env, capture_output=True, text=True,
            check=check, timeout=120,
        )

    def _ensure_git_repo(self) -> None:
        repo = self.shadow / "repo.git"
        if not repo.exists():
            repo.mkdir(parents=True, exist_ok=True)
            self._git("init", "--bare", str(repo))
            excludes = repo / "info" / "exclude"
            excludes.parent.mkdir(parents=True, exist_ok=True)
            excludes.write_text(
                "\n".join(self.ignores) + "\n", encoding="utf-8"
            )

    # -- public API --------------------------------------------------------

    def checkpoint(self, label: str = "", memory_turns: int = 0) -> Checkpoint:
        """Snapshot the workspace now. Cheap no-op commit if nothing changed."""
        if self.backend == "git":
            self._ensure_git_repo()
            self._git("add", "-A")
            msg = label or f"checkpoint {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._git("commit", "--allow-empty", "-m", msg)
            cid = self._git("rev-parse", "HEAD").stdout.strip()
        else:
            cid = uuid.uuid4().hex[:12]
            dest = self.shadow / "snapshots" / cid
            shutil.copytree(
                self.workspace, dest,
                ignore=shutil.ignore_patterns(*self.ignores),
            )
        cp = Checkpoint(
            checkpoint_id=cid, label=label, created_ts=time.time(),
            memory_turns=memory_turns, backend=self.backend,
        )
        self.checkpoints.append(cp)
        logger.info("Checkpoint %s created (%s backend).", cid[:8], cp.backend)
        return cp

    def restore(
        self,
        checkpoint_id: str,
        mode: str = "both",
        memory: Optional[object] = None,
    ) -> dict:
        """Restore to *checkpoint_id*. ``mode``: ``files`` | ``task`` | ``both``.

        ``task``/``both`` require the session's ConversationMemory to be
        passed so its turns can be truncated to the recorded count.
        """
        if mode not in ("files", "task", "both"):
            raise ValueError("mode must be 'files', 'task', or 'both'")
        cp = next(
            (c for c in self.checkpoints
             if c.checkpoint_id.startswith(checkpoint_id)), None,
        )
        if cp is None:
            raise ValueError(f"Unknown checkpoint '{checkpoint_id}'.")

        outcome: dict = {"checkpoint_id": cp.checkpoint_id, "mode": mode}
        if mode in ("files", "both"):
            if cp.backend == "git":
                # restore tracked content, then delete files that did not
                # exist at the checkpoint (rm cached diff)
                self._git("restore", "--source", cp.checkpoint_id,
                          "--worktree", "--", ".")
                current = set(
                    self._git("ls-files", "--others", "--exclude-standard")
                    .stdout.splitlines()
                )
                tracked_now = set(self._git("ls-files").stdout.splitlines())
                tracked_then = set(
                    self._git("ls-tree", "-r", "--name-only", cp.checkpoint_id)
                    .stdout.splitlines()
                )
                extras = (tracked_now | current) - tracked_then
                for rel in extras:
                    target = self.workspace / rel
                    if target.is_file():
                        target.unlink()
                outcome["files_deleted"] = sorted(extras)
            else:
                src = self.shadow / "snapshots" / cp.checkpoint_id
                shutil.copytree(src, self.workspace, dirs_exist_ok=True)
                outcome["files_deleted"] = []  # copy backend limitation
            outcome["files_restored"] = True
        if mode in ("task", "both"):
            if memory is None or not hasattr(memory, "truncate_to"):
                raise ValueError(
                    "task restore needs the session's ConversationMemory."
                )
            outcome["turns_removed"] = memory.truncate_to(cp.memory_turns)
        logger.info("Restored checkpoint %s (mode=%s).", cp.checkpoint_id[:8], mode)
        return outcome
