"""Record every LLM exchange so the loop can be audited after the fact.

The system's claim is that the search learns from measured feedback. That is
only checkable if you can see what the model was actually shown and what it did
with it -- and this session found several defects that were invisible from the
outside: format specs shipped where numbers belonged, a horizon instruction that
contradicted the gate, seven tear-sheet metrics that reached zero prompts, and
context blocks that vanished silently when their loader failed.

None of those show up in a metric. All of them are obvious in the transcript.

Written as JSONL, one record per call, with the call site resolved from the
stack so a record can be traced to the operator that made it. Enabled by
default; set QA_LLM_LOG=0 to disable, QA_LLM_LOG_DIR to relocate.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_INSTALLED = False

# Frames belonging to the logging or client machinery itself -- skipped when
# resolving which operator made the call.
_SKIP = ("llm/client.py", "llm/io_log.py", "functools.py")


def enabled() -> bool:
    return os.environ.get("QA_LLM_LOG", "1").strip().lower() not in ("0", "false", "no")


def log_path() -> Path:
    d = Path(os.environ.get("QA_LLM_LOG_DIR", "logs/llm"))
    d.mkdir(parents=True, exist_ok=True)
    # run.sh sets EXPERIMENT_ID; accept either spelling. Without the second,
    # the log falls back to a date-only name and two runs on the same day append
    # to one file, which makes the per-run audit read as one incoherent run.
    run = (os.environ.get("QA_EXPERIMENT_ID")
           or os.environ.get("EXPERIMENT_ID")
           or time.strftime("%Y%m%d"))
    return d / f"llm_io_{run}.jsonl"


def _caller() -> str:
    """Which operator made this call -- refine, crossover, diagnosis, planning."""
    for frame in inspect.stack()[2:]:
        fn = frame.filename
        if any(s in fn for s in _SKIP):
            continue
        if "quantaalpha" not in fn:
            continue
        return f"{Path(fn).parent.name}/{Path(fn).name}:{frame.lineno} {frame.function}"
    return "unknown"


def _record(**kw) -> None:
    try:
        with _LOCK, log_path().open("a") as fh:
            fh.write(json.dumps(kw, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass          # logging must never break a run


def install() -> bool:
    """Wrap the client's chat entry point. Idempotent."""
    global _INSTALLED
    if _INSTALLED or not enabled():
        return False
    from quantaalpha.llm import client as _client

    target = _client.APIBackend.build_messages_and_create_chat_completion
    if getattr(target, "_qa_io_logged", False):
        _INSTALLED = True
        return False

    @functools.wraps(target)
    def wrapper(self, user_prompt, system_prompt=None, *a, **kw):
        site = _caller()
        t0 = time.time()
        err = None
        out = None
        try:
            out = target(self, user_prompt, system_prompt, *a, **kw)
            return out
        except Exception as exc:                       # record failures too --
            err = f"{type(exc).__name__}: {exc}"       # a silent LLM failure is
            raise                                      # exactly what hides bugs
        finally:
            up = user_prompt or ""
            sp = system_prompt or ""
            _record(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                site=site,
                seconds=round(time.time() - t0, 2),
                json_mode=bool(kw.get("json_mode", False)),
                system_prompt=sp,
                user_prompt=up,
                response=out,
                error=err,
                # Cheap integrity handles: a prompt that stops changing between
                # rounds means the feedback channel is dead, and identical
                # hashes across a run are the signature of that.
                prompt_chars=len(up) + len(sp),
                prompt_sha=hashlib.sha256((sp + "\x00" + up).encode()).hexdigest()[:12],
                response_chars=len(out or ""),
            )

    wrapper._qa_io_logged = True
    _client.APIBackend.build_messages_and_create_chat_completion = wrapper
    _INSTALLED = True
    return True


__all__ = ["install", "enabled", "log_path"]
