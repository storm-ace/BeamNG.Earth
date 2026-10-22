"""The rebuild: the kunstwerk tools as subprocesses, one job at a time, their output kept for the page's event stream.

    placement (optional)  tools/kunstwerk_placement.py <ref>            -> ref/placement.json
    solid                 tools/kunstwerk_solid.py <ref>                -> parts.json, furniture.json (overrides applied)
    export                tools/kunstwerk_export.py <kw> --name <name>  -> server/data/kunstwerken/<name>.kunstwerk.json
"""
import os
import subprocess
import threading
import time
import uuid


def default_steps(cfg, placement=False):
    py, tools = cfg.python, os.path.join(cfg.repo, "tools")
    steps = []
    if placement:
        steps.append(("plaatsing", [py, "-u", os.path.join(tools, "kunstwerk_placement.py"), cfg.ref]))
    steps.append(("model", [py, "-u", os.path.join(tools, "kunstwerk_solid.py"), cfg.ref]))
    steps.append(("export", [py, "-u", os.path.join(tools, "kunstwerk_export.py"), cfg.kw_dir, "--name", cfg.name,
                             "--out", os.path.dirname(cfg.mesh_path)]))
    return steps


class Job:
    def __init__(self, steps, cwd):
        self.id = uuid.uuid4().hex[:12]
        self.steps, self.cwd = steps, cwd
        self.lines, self.done, self.ok = [], False, None
        self.started = time.time()
        self.cond = threading.Condition()

    def emit(self, line):
        with self.cond:
            self.lines.append(line.rstrip("\r\n"))
            self.cond.notify_all()

    def run(self):
        ok = True
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            for name, cmd in self.steps:
                self.emit(f"== {name}: {' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd))}")
                t0 = time.time()
                p = subprocess.Popen(cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in p.stdout:
                    self.emit(line)
                rc = p.wait()
                self.emit(f"== {name}: exit {rc} ({time.time() - t0:.1f} s)")
                if rc != 0:
                    ok = False
                    break
        except Exception as e:          # (a missing interpreter, a bad path: shown on the page)
            self.emit(f"!! {type(e).__name__}: {e}")
            ok = False
        with self.cond:
            self.ok, self.done = ok, True
            self.cond.notify_all()

    def wait_lines(self, start, timeout=1.0):
        """Lines from index start on (waits up to timeout for new ones); (lines, done)."""
        with self.cond:
            if len(self.lines) <= start and not self.done:
                self.cond.wait(timeout)
            return self.lines[start:], self.done


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.current = None

    def start(self, steps, cwd):
        """A new job, or None while one runs."""
        with self.lock:
            if self.current is not None and not self.current.done:
                return None
            job = Job(steps, cwd)
            self.jobs[job.id] = job
            self.current = job
        threading.Thread(target=job.run, daemon=True).start()
        return job
