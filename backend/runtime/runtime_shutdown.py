from __future__ import annotations
import os
import signal
import time
from pathlib import Path
from backend.runtime.runtime_process import RuntimeProcess


class RuntimeShutdownError(Exception):
    """Raised when runtime shutdown fails."""


class RuntimeShutdown:
    """
    Gracefully terminates heavyweight inference runtimes.
    Supports:
    - process-group termination
    - graceful SIGTERM
    - forced SIGKILL escalation
    """

    DEFAULT_SHUTDOWN_WAIT = 2.0
    DEFAULT_POST_KILL_WAIT = 5.0

    def __init__(
        self,
        *,
        shutdown_wait: float = DEFAULT_SHUTDOWN_WAIT,
        post_kill_wait: float = DEFAULT_POST_KILL_WAIT,
        escalate_to_sigkill: bool = True,
    ) -> None:
        self.shutdown_wait = shutdown_wait
        self.post_kill_wait = post_kill_wait
        self.escalate_to_sigkill = escalate_to_sigkill

    def shutdown(self, process: RuntimeProcess) -> None:
        if process.pid is None:
            return
        if not process.active:
            return

        child_pids = self._descendant_pids(process.pid)

        try:
            self._terminate_group(process, signal.SIGTERM, child_pids)
        except ProcessLookupError:
            process.mark_stopped()
            return
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid}"
            ) from exc

        self._wait_until_dead(process, child_pids, timeout=self.shutdown_wait)

        if self.escalate_to_sigkill and self._any_alive(process, child_pids):
            try:
                child_pids = self._descendant_pids(process.pid) or child_pids
                self._terminate_group(process, signal.SIGKILL, child_pids)
            except ProcessLookupError:
                pass
            self._wait_until_dead(process, child_pids, timeout=self.post_kill_wait)

        process.mark_stopped()

    def shutdown_nowait(self, process: RuntimeProcess) -> None:
        if process.pid is None or not process.active:
            return

        try:
            self._terminate_group(
                process,
                signal.SIGTERM,
                self._descendant_pids(process.pid),
            )
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid}"
            ) from exc
        finally:
            process.mark_stopped()

    def is_alive(self, process: RuntimeProcess) -> bool:
        if process.pid is None:
            return False
        try:
            if process.pgid is not None:
                os.killpg(process.pgid, 0)
            else:
                os.kill(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _terminate_group(
        self,
        process: RuntimeProcess,
        sig: signal.Signals,
        child_pids: set[int] | None = None,
    ) -> None:
        """
        Kill entire vLLM process group.
        This is REQUIRED because vLLM spawns child workers.
        """
        if process.pgid is not None:
            os.killpg(process.pgid, sig)
        else:
            os.kill(process.pid, sig)

        # Some worker launch paths can escape the original process group.
        for pid in child_pids or set():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _wait_until_dead(
        self,
        process: RuntimeProcess,
        child_pids: set[int],
        *,
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._reap_pid(process.pid)
            if not self._any_alive(process, child_pids):
                return True
            time.sleep(0.2)
        return not self._any_alive(process, child_pids)

    def _any_alive(self, process: RuntimeProcess, child_pids: set[int]) -> bool:
        return self.is_alive(process) or any(
            self._pid_alive(pid) for pid in child_pids
        )

    @staticmethod
    def _reap_pid(pid: int | None) -> None:
        if pid is None:
            return
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        except ProcessLookupError:
            return

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _descendant_pids(self, root_pid: int) -> set[int]:
        children_by_parent: dict[int, set[int]] = {}
        proc_root = Path("/proc")
        if not proc_root.exists():
            return set()

        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                parent_pid = self._parent_pid_from_stat(stat)
            except (OSError, ValueError):
                continue
            children_by_parent.setdefault(parent_pid, set()).add(int(entry.name))

        descendants: set[int] = set()
        pending = list(children_by_parent.get(root_pid, set()))
        while pending:
            pid = pending.pop()
            if pid in descendants:
                continue
            descendants.add(pid)
            pending.extend(children_by_parent.get(pid, set()))
        return descendants

    @staticmethod
    def _parent_pid_from_stat(stat: str) -> int:
        # /proc/<pid>/stat wraps process names in parentheses; ppid is after them.
        after_name = stat.rsplit(")", 1)[1].strip()
        fields = after_name.split()
        return int(fields[1])
