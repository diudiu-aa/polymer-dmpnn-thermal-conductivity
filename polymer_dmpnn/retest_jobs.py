"""One cancellable, loopback UI retest at a time, with disposable artifacts."""
from __future__ import annotations
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile


class RetestJobs:
    def __init__(self, reproduction_directory, runner='retest_models.py', timeout_seconds=600):
        self.directory = reproduction_directory
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.available = self.directory is not None and all(
            (self.directory / p).is_file() for p in (runner, 'inputs/splits.json', 'inputs/records.csv.gz'))
        self.temp = None
        self.job = None
        self.lock = threading.Lock()

    def _refresh(self):
        job = self.job
        if job is None:
            return
        process = job['process']
        if process.poll() is None and time.monotonic() - job['clock'] > self.timeout_seconds:
            self._stop('timeout')
        code = process.poll()
        progress_file = job['output'] / 'progress.json'
        if progress_file.exists():
            try:
                job['progress'] = json.loads(progress_file.read_text())
            except (ValueError, OSError):
                pass  # Atomic replacement will be visible on the next poll.
        if code is not None:
            job['log'].close()
            if job.get('forced_state'):
                job['state'] = job['forced_state']
            elif (job['output'] / 'result.json').exists() and code in (0, 2):
                job['state'] = 'completed' if code == 0 else 'different'
            else:
                job['state'] = 'failed'
            job['returncode'] = code

    def start(self, options=None):
        with self.lock:
            if not self.available:
                raise ValueError('未找到真实重测所需文件。请下载完整GitHub仓库并从该目录启动。')
            self._refresh()
            if self.job is not None and self.job['process'].poll() is None:
                return {'run_id': self.job['id'], 'state': 'running', 'already_running': True}
            extra = []
            if self.runner == 'train_from_scratch.py':
                mode = (options or {}).get('mode', 'quick')
                if mode not in ('quick', 'full'):
                    raise ValueError('请选择配对试跑或完整5组重复。')
                extra = ['--repeats', '1' if mode == 'quick' else '5']
            # Keep just the latest run; the user can download it before restarting.
            if self.temp is not None:
                self.temp.cleanup()
            self.temp = tempfile.TemporaryDirectory(prefix='polymer-model-retest-')
            root = Path(self.temp.name)
            run_id = uuid.uuid4().hex
            output = root / ('run-' + run_id)
            log = (root / 'run.log').open('w')
            try:
                process = subprocess.Popen([sys.executable, '-I', str(self.directory / self.runner),
                                            '--output', str(output), *extra], cwd=self.directory.parent,
                                           stdout=log, stderr=subprocess.STDOUT)
            except Exception:
                log.close();self.temp.cleanup();self.temp=None
                raise
            self.job = {'id': run_id, 'output': output, 'log_path': root / 'run.log', 'log': log,
                        'process': process, 'clock': time.monotonic(), 'state': 'running',
                        'started_at': datetime.now(timezone.utc).isoformat(), 'progress': {}}
            return {'run_id': run_id, 'state': 'running', 'already_running': False}

    def status(self, run_id=None):
        with self.lock:
            self._refresh()
            if self.job is None:
                return {'state': 'idle', 'available': self.available}
            job = self.job
            if run_id is not None and run_id != job['id']:
                raise ValueError('这次运行已结束保留，请查看当前运行记录。')
            result = {'run_id': job['id'], 'state': job['state'], 'started_at': job['started_at'],
                      'elapsed_seconds': time.monotonic() - job['clock'], 'progress': job['progress']}
            if job['state'] in ('completed', 'different'):
                result['result'] = json.loads((job['output'] / 'result.json').read_text())
                result['elapsed_seconds'] = result['result']['elapsed_seconds']
            elif job['state'] in ('failed', 'timeout'):
                result['error'] = job['progress'].get('error', '运行未完成，请下载日志查看原因。')
                result['log_tail'] = job['log_path'].read_text(errors='replace')[-3000:]
            return result

    def _stop(self, reason):
        job = self.job
        if job is None or job['process'].poll() is not None:
            return
        job['forced_state'] = reason
        job['process'].terminate()
        try:
            job['process'].wait(timeout=3)
        except subprocess.TimeoutExpired:
            job['process'].kill();job['process'].wait(timeout=3)

    def cancel(self, run_id):
        with self.lock:
            if self.job is None or self.job['id'] != run_id:
                raise ValueError('没有这次运行。')
            self._stop('cancelled');self._refresh()
            return {'run_id': run_id, 'state': self.job['state']}

    def archive(self, run_id):
        with self.lock:
            self._refresh()
            if self.job is None or self.job['id'] != run_id:
                raise ValueError('没有这次运行。')
            if self.job['process'].poll() is None:
                raise ValueError('模型仍在计算，请完成或停止后下载。')
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(self.job['log_path'], 'run.log')
                z.writestr('web_run.json', json.dumps({'run_id': run_id, 'started_at': self.job['started_at'],
                    'state': self.job['state'], 'returncode': self.job.get('returncode')}, indent=2))
                if self.job['output'].exists():
                    for f in sorted(self.job['output'].rglob('*')):
                        if f.is_file() and f.suffix in ('.json', '.csv', '.pt'):
                            z.write(f, str(f.relative_to(self.job['output'])))
            return buffer.getvalue()

    def close(self):
        with self.lock:
            self._stop('cancelled')
            if self.job is not None:
                self.job['log'].close()
            if self.temp is not None:
                self.temp.cleanup();self.temp=None
