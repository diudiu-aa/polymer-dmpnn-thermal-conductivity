"""Loopback-only browser interface for the published models; no training."""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata, resources
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from .retest_jobs import RetestJobs

MAX_ROWS = 1000
MAX_BODY = 2 * 1024 * 1024
MAX_SMILES = 2000
MODEL_LOCK = threading.Lock()


def read_csv(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('CSV内容为空，请上传文件或粘贴CSV。')
    if '\ufffd' in text:
        raise ValueError('CSV编码无法识别，请另存为UTF-8 CSV再上传。')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff'), newline=''))
    fields = reader.fieldnames or []
    if not fields or len(fields) != len(set(fields)) or any(not f.strip() for f in fields):
        raise ValueError('CSV需要第一行列名，且列名不能重复或留空。')
    rows = []
    try:
        for row in reader:
            if None in row:
                raise ValueError('CSV某行的列数超过表头，请检查逗号与引号。')
            if all(v is None or not v.strip() for v in row.values()):
                continue
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f'每批最多{MAX_ROWS}条记录，请拆分CSV后再测试。')
    except csv.Error as exc:
        raise ValueError('CSV格式无效，请检查引号与分隔符。') from exc
    if not rows:
        raise ValueError('CSV只有表头，没有可测试记录。')
    return fields, rows


def local_error(text):
    translations = {
        'SMILES is empty': '结构为空，请填写重复单元SMILES。',
        'invalid repeat-unit SMILES': '结构无法解析，请检查SMILES写法。',
        'expected exactly two degree-one polymerization wildcards': '重复单元必须恰好有两个末端连接点 *。',
        'repeat unit must be a single connected molecule': '请使用一个连通的重复单元，不支持用点号拼接多个分子。',
        'isotopic or mapped wildcard endpoints need an explicit mapping rule': '当前连接点不支持同位素或原子映射编号，请使用普通 *。',
        'the two polymerization endpoints must both specify single bonds': '两个末端 * 都需要通过单键连接。',
    }
    return translations.get(text, '该结构暂时无法计算：' + text)


def model_options():
    from .paper import paper_registry
    registry = paper_registry()['models']
    arms = {'open': '开链', 'periodic': '周期', 'periodic_rank': '周期 + 排序目标',
            'balanced_control': '周期 + 平衡采样', 'multitask': '周期 + 双任务'}
    tables = {'table3': '第一套结构簇对照', 'table4': '学习曲线', 'table5': '第二套结构簇对照'}
    return [{'id': key, 'label': f"{tables[s['experiment']]} · {arms[s['arm']]} · {s['fit_budget']}标签 · 种子{s['seed']}",
             **{k: s[k] for k in ('experiment', 'arm', 'seed', 'fit_budget')}}
            for key, s in sorted(registry.items())]


@lru_cache(maxsize=4)
def load_model(model_id):
    import torch
    torch.set_num_threads(1)
    if model_id == 'legacy_ensemble':
        from .predictor import PolymerThermalPredictor
        return PolymerThermalPredictor()
    from .paper import PaperPredictor
    return PaperPredictor(model_id)


def structure_svg(smiles):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 240)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText().encode()).decode()


def validate_model_id(model_id):
    if model_id != 'legacy_ensemble' and model_id not in {s['id'] for s in model_options()}:
        raise ValueError('请选择列表中已发布的模型。')


def score_references(rows):
    pairs = [(r['reference_kappa_W_mK'], r['predicted_kappa_W_mK']) for r in rows
             if r.get('success') and r.get('reference_kappa_W_mK') is not None]
    if not pairs:
        return None
    errors = [p - y for y, p in pairs]
    ys = [y for y, _ in pairs]
    mean = math.fsum(ys) / len(ys)
    denominator = math.fsum((y - mean) ** 2 for y in ys)
    return {'n': len(pairs), 'mae': math.fsum(abs(e) for e in errors) / len(errors),
            'rmse': math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
            'r2': 1 - math.fsum(e * e for e in errors) / denominator if len(pairs) > 1 and denominator > 0 else None}


def predict_request(payload):
    mode = payload.get('mode', 'single')
    model_id = payload.get('model_id', 'table5_B600_periodic_32101')
    validate_model_id(model_id)
    reference_column = None
    if mode == 'single':
        inputs = [{'input_row': 1, 'input_id': '', 'smiles': payload.get('smiles', '')}]
    elif mode == 'csv':
        fields, raw = read_csv(payload.get('csv_text'))
        column = payload.get('smiles_column', 'smiles')
        reference_column = payload.get('reference_column') or None
        if column not in fields:
            raise ValueError('请从CSV列名中选择结构列。')
        if reference_column is not None and (reference_column not in fields or reference_column == column):
            raise ValueError('参考热导率列需要与结构列不同。')
        inputs = [{'input_row': i, 'input_id': r.get('id', '') or '', 'smiles': r.get(column) or '',
                   '_reference': r.get(reference_column) if reference_column else None}
                  for i, r in enumerate(raw, 1)]
    else:
        raise ValueError('未识别的输入方式。')
    if any(not isinstance(r['smiles'], str) or len(r['smiles']) > MAX_SMILES for r in inputs):
        raise ValueError(f'单条SMILES不能超过{MAX_SMILES}个字符。')
    start = time.monotonic()
    rows = []
    with MODEL_LOCK:
        predictor = load_model(model_id)
        for item in inputs:
            result = predictor.predict_one(item['smiles'])
            if not result['success']:
                result['error'] = local_error(result['error'])
            row = {'input_row': item['input_row'], 'input_id': item['input_id'],
                   **result, 'model_id': model_id, 'reference_kappa_W_mK': None}
            if reference_column:
                try:
                    value = float(item['_reference'])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError
                    row['reference_kappa_W_mK'] = value
                    if result['success']:
                        row['prediction_minus_reference_W_mK'] = result['predicted_kappa_W_mK'] - value
                except (ValueError, TypeError):
                    row['reference_error'] = '该行参考值为空或不是正的有限数，未计入误差指标。'
            rows.append(row)
    svg = None
    if mode == 'single' and rows[0]['success']:
        try:
            svg = structure_svg(rows[0]['smiles'])
        except Exception:
            # A depiction failure must not erase a successful model prediction.
            svg = None
    from rdkit import rdBase
    import torch
    import numpy
    try:
        version = metadata.version('polymer-dmpnn-kappa')
    except metadata.PackageNotFoundError:
        version = 'source checkout'
    return {'rows': rows, 'summary': {'total': len(rows), 'success': sum(r['success'] for r in rows),
             'failed': sum(not r['success'] for r in rows),
             'reference_skipped': sum(bool(r.get('reference_error')) for r in rows),
             'elapsed_seconds': time.monotonic() - start}, 'metrics': score_references(rows),
            'structure_svg_base64': svg, 'metadata': {'model_id': model_id, 'unit': 'W/(m K)',
             'target': '300 K / 1 atm / nominally unstrained amorphous bulk MD reference',
             'package_version': version, 'rdkit': rdBase.rdkitVersion, 'torch': torch.__version__,
             'numpy': numpy.__version__, 'statistics': 'One selected model or legacy ensemble; not the paper mean of per-seed metrics.'}}


def find_reproduction_root(root=None):
    candidates = [Path(root)] if root is not None else [Path.cwd(), Path(__file__).resolve().parents[1]]
    for candidate in candidates:
        directory = candidate.resolve() / 'reproduce'
        if all((directory / p).is_file() for p in ('paper_tables.py', 'data/manifest.json', 'data/saved_predictions.csv.gz')):
            return directory
    return None


def replay_tables(directory):
    if directory is None:
        raise ValueError('未找到论文复算数据。请下载完整GitHub仓库，在仓库根目录启动界面。')
    with tempfile.TemporaryDirectory(prefix='polymer-ui-tables-') as temp:
        output = Path(temp) / 'tables'
        proc = subprocess.run([sys.executable, '-I', str(directory / 'paper_tables.py'), '--verify', '--output', str(output)],
                              capture_output=True, text=True, timeout=60, check=False)
        if proc.returncode:
            raise RuntimeError('复算未通过，请核对下载包是否完整、数据文件是否被修改。')
        verification = json.loads((output / 'verification.json').read_text())
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(output.iterdir()):
                z.write(f, f.name)
        tables = {f'table_{n}.csv': (output / f'table_{n}.csv').read_text() for n in range(1, 6)}
        table_data = {}
        for name, contents in tables.items():
            records = list(csv.reader(io.StringIO(contents)))
            table_data[name] = {'columns': records[0], 'rows': records[1:]}
        return {'verification': verification, 'zip_base64': base64.b64encode(archive.getvalue()).decode(),
                'tables': tables, 'table_data': table_data}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self):
        if hasattr(self, 'retests'):
            self.retests.close()
        super().server_close()


def make_server(port=0, repository=None):
    directory = find_reproduction_root(repository)
    index = resources.files('polymer_dmpnn').joinpath('web_assets/index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, content, content_type='application/json; charset=utf-8'):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)

        def local_request(self):
            port = self.server.server_address[1]
            allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            return host in allowed and (origin is None or origin in {f'http://{h}' for h in allowed})

        def do_GET(self):
            if not self.local_request():
                return self.reply(403, {'error': '仅接受本机页面请求。'})
            route = urlsplit(self.path).path
            if route == '/':
                return self.reply(200, index, 'text/html; charset=utf-8')
            if route == '/api/info':
                return self.reply(200, {'models': model_options(), 'reproduction_available': directory is not None,
                                       'retest_available': self.server.retests.available, 'max_rows': MAX_ROWS})
            if route in ('/api/retest/status', '/api/retest/download'):
                try:
                    run_id = parse_qs(urlsplit(self.path).query).get('run_id', [None])[0]
                    if route.endswith('/download'):
                        return self.reply(200, self.server.retests.archive(run_id), 'application/zip')
                    return self.reply(200, self.server.retests.status(run_id))
                except ValueError as exc:
                    return self.reply(400, {'error': str(exc)})
            if route == '/api/example.csv':
                return self.reply(200, b'id,smiles\npolyethylene,*CC*\npolypropylene,*CC(*)C\npolystyrene,*CC(*)c1ccccc1\n', 'text/csv; charset=utf-8')
            self.reply(404, {'error': '未找到该页面。'})

        def do_POST(self):
            if not self.local_request():
                return self.reply(403, {'error': '仅接受本机页面请求。'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > MAX_BODY:
                    return self.reply(413, {'error': '输入为空或超过2 MB，请缩小文件后重试。'})
                payload = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('请求格式无效。')
                route = urlsplit(self.path).path
                if route == '/api/predict':
                    return self.reply(200, predict_request(payload))
                if route == '/api/inspect':
                    fields, rows = read_csv(payload.get('csv_text'))
                    suggested = next((f for f in fields if f.lower().strip() in ['smiles', 'psmiles']), fields[0])
                    return self.reply(200, {'columns': fields, 'rows': len(rows), 'smiles_column': suggested})
                if route == '/api/reproduce':
                    return self.reply(200, replay_tables(directory))
                if route == '/api/retest/start':
                    return self.reply(202, self.server.retests.start())
                if route == '/api/retest/cancel':
                    return self.reply(200, self.server.retests.cancel(payload.get('run_id')))
                self.reply(404, {'error': '未找到该操作。'})
            except (ValueError, UnicodeError, csv.Error) as exc:
                self.reply(400, {'error': str(exc)})
            except subprocess.TimeoutExpired:
                self.reply(500, {'error': '复算超过60秒，已停止。请用命令行检查数据包。'})
            except Exception as exc:
                print(f'Local UI error: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                self.reply(500, {'error': '运行未完成。请检查安装包、权重与输入，或将终端信息反馈给维护者。'})
    server = LocalServer(('127.0.0.1', port), Handler)
    server.retests = RetestJobs(directory)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open a local browser interface for polymer thermal-conductivity prediction.')
    parser.add_argument('--port', type=int, default=0, help='Local port; 0 selects an available port')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    parser.add_argument('--repository', type=Path, help='Downloaded repository root for table reproduction')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535')
    server = make_server(args.port, args.repository)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'聚合物热导率界面已启动：{url}\n仅在本机运行。按 Ctrl+C 关闭。', flush=True)
    if not args.no_browser:
        threading.Timer(.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
