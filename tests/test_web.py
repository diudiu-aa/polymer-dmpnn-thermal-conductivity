"""Exercise the local browser API with real published weights and public data."""
import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile

from polymer_dmpnn.paper import PaperPredictor
from polymer_dmpnn.web import make_server, read_csv


class LocalWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.server = make_server(0, cls.repo)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def request(self, path, data=None, headers=None):
        payload = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, data=payload,
                                     headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                raw = response.read()
                return response.status, raw, dict(response.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def test_home_and_model_catalog(self):
        code, raw, headers = self.request('/')
        self.assertEqual(code, 200)
        self.assertIn('运行论文主结果', raw.decode())
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        code, raw, _ = self.request('/api/info')
        info = json.loads(raw)
        self.assertEqual(len(info['models']), 65)
        self.assertTrue(info['reproduction_available'])

    def test_actual_single_prediction_matches_selected_checkpoint(self):
        model = 'table5_B600_periodic_32101'
        code, raw, _ = self.request('/api/predict', {'mode': 'single', 'smiles': '*CC*', 'model_id': model})
        result = json.loads(raw)
        self.assertEqual(code, 200, result)
        expected = PaperPredictor(model).predict_one('*CC*')
        self.assertEqual(result['rows'][0]['predicted_kappa_W_mK'], expected['predicted_kappa_W_mK'])
        self.assertEqual(result['metadata']['model_id'], model)
        self.assertEqual(result['summary']['success'], 1)
        self.assertIn(b'<svg', base64.b64decode(result['structure_svg_base64']))

    def test_multitask_output_preserves_classifier_logit(self):
        model = 'table5_B600_multitask_32101'
        code, raw, _ = self.request('/api/predict', {'mode': 'single', 'smiles': '*CC*', 'model_id': model})
        result = json.loads(raw)
        self.assertEqual(code, 200)
        expected = PaperPredictor(model).predict_one('*CC*')
        self.assertEqual(result['rows'][0]['high_kappa_classification_logit'], expected['high_kappa_classification_logit'])

    def test_batch_preserves_failed_rows_and_scores_only_valid_pairs(self):
        code, raw, _ = self.request('/api/predict', {'mode': 'csv', 'model_id': 'table5_B600_periodic_32101',
            'csv_text': 'id,structure,kappa\na,*CC*,0.30\nb,CC,0.20\nc,*CCO*,0.25\nd,*CC(*)C,invalid\n',
            'smiles_column': 'structure', 'reference_column': 'kappa'})
        r = json.loads(raw)
        self.assertEqual(code, 200, r)
        self.assertEqual([x['input_id'] for x in r['rows']], ['a', 'b', 'c', 'd'])
        self.assertEqual(r['summary']['failed'], 1)
        self.assertIn('两个末端连接点', r['rows'][1]['error'])
        self.assertEqual(r['summary']['reference_skipped'], 1)
        self.assertEqual(r['metrics']['n'], 2)
        a, c = r['rows'][0], r['rows'][2]
        errors = [a['predicted_kappa_W_mK'] - .3, c['predicted_kappa_W_mK'] - .25]
        self.assertAlmostEqual(r['metrics']['mae'], sum(abs(e) for e in errors) / 2, places=14)
        self.assertAlmostEqual(r['metrics']['r2'], 1 - sum(e*e for e in errors)/(.025**2*2), places=12)

    def test_csv_header_and_bom_detection(self):
        code, raw, _ = self.request('/api/inspect', {'csv_text': '\ufeffid,SMILES\na,*CC*\n'})
        r = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertEqual(r['smiles_column'], 'SMILES')
        self.assertEqual(r['rows'], 1)
        with self.assertRaises(ValueError):
            read_csv('smiles,smiles\n*CC*,*CC*')
        with self.assertRaises(ValueError):
            read_csv('smiles\n' + '*CC*\n' * 1001)

    def test_invalid_requests_report_actionable_errors(self):
        for payload in [{'mode': 'single', 'smiles': '*CC*', 'model_id': 'not-a-model'},
                        {'mode': 'csv', 'csv_text': 'wrong\n*CC*\n', 'smiles_column': 'smiles'}]:
            code, raw, _ = self.request('/api/predict', payload)
            self.assertEqual(code, 400, raw)
            self.assertTrue(json.loads(raw)['error'])
        code, raw, _ = self.request('/api/predict', {'smiles': '*CC*'}, {'Origin': 'https://example.com'})
        self.assertEqual(code, 403)
        code, _, _ = self.request('/not-found')
        self.assertEqual(code, 404)

    def test_reproduction_download_contains_verified_tables(self):
        code, raw, _ = self.request('/api/reproduce', {})
        r = json.loads(raw)
        self.assertEqual(code, 200, r)
        self.assertEqual(r['verification']['verification']['display_statistics_checked'], 182)
        self.assertEqual(r['verification']['verification']['display_mismatches'], 0)
        self.assertEqual(len(r['table_data']), 5)
        for name, contents in r['tables'].items():
            parsed = list(csv.reader(io.StringIO(contents)))
            self.assertEqual(r['table_data'][name]['columns'], parsed[0])
            self.assertEqual(r['table_data'][name]['rows'], parsed[1:])
        # The plotted second-split tail result must remain its actual CSV value.
        t5 = r['table_data']['table_5.csv']
        tail_column = t5['columns'].index('Tail MAE (W m⁻¹ K⁻¹)')
        multitask = next(row for row in t5['rows'] if row[0] == 'Periodic + multitask, regression ranking')
        self.assertEqual(multitask[tail_column], '0.04902 ± 0.00700')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(r['zip_base64']))) as z:
            for n in range(1, 6):
                self.assertIn(f'table_{n}.csv', z.namelist())
            self.assertIn('verification.json', z.namelist())

    def test_cancel_retest_stops_the_computing_process(self):
        code, raw, _ = self.request('/api/retest/start', {})
        run_id = json.loads(raw)['run_id']
        self.assertEqual(code, 202)
        code, raw, _ = self.request('/api/retest/cancel', {'run_id': run_id})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['state'], 'cancelled')
        self.assertIsNotNone(self.server.retests.job['process'].poll())
        code, raw, _ = self.request('/api/retest/status?run_id=' + run_id)
        self.assertEqual(json.loads(raw)['state'], 'cancelled')

    def test_training_start_stop_and_compute_jobs_are_mutually_exclusive(self):
        code, raw, _ = self.request('/api/training/start', {'mode': 'unknown'})
        self.assertEqual(code, 400)
        code, raw, _ = self.request('/api/training/start', {'mode': 'quick'})
        self.assertEqual(code, 202, raw)
        run_id = json.loads(raw)['run_id']
        code, raw, _ = self.request('/api/retest/start', {})
        self.assertEqual(code, 400)
        code, raw, _ = self.request('/api/training/status?run_id=' + run_id)
        self.assertEqual(json.loads(raw)['state'], 'running')
        code, raw, _ = self.request('/api/training/cancel', {'run_id': run_id})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['state'], 'cancelled')
        self.assertIsNotNone(self.server.training.job['process'].poll())
        code, raw, _ = self.request('/api/training/download?run_id=' + run_id)
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertIn('run.log', z.namelist())

    def test_fresh_retest_actually_runs_models_and_downloads_new_predictions(self):
        code, raw, _ = self.request('/api/retest/start', {})
        self.assertEqual(code, 202, raw)
        started = json.loads(raw)
        run_id = started['run_id']
        code, raw, _ = self.request('/api/retest/start', {})
        self.assertEqual(json.loads(raw)['run_id'], run_id)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            code, raw, _ = self.request('/api/retest/status?run_id=' + run_id)
            status = json.loads(raw)
            if status['state'] != 'running':
                break
            time.sleep(.1)
        self.assertEqual(status['state'], 'completed', status)
        result = status['result']
        self.assertEqual(result['summary']['groups_tested'], 15)
        self.assertEqual(result['summary']['predictions_tested'], 6000)
        self.assertEqual(result['summary']['forward_batches'], 375)
        self.assertTrue(result['summary']['prediction_tolerance_pass'])
        self.assertEqual(result['summary']['display_statistics_checked'], 34)
        self.assertEqual(len(result['receipt']['models']), 15)
        self.assertIn('before stored predictions were loaded', result['receipt']['phase'])
        code, archive, _ = self.request('/api/retest/download?run_id=' + run_id)
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            self.assertIn('fresh_prediction_receipt.json', z.namelist())
            self.assertIn('candidate_rankings.csv', z.namelist())
            self.assertIn('metric_comparison.csv', z.namelist())
            self.assertEqual(json.loads(z.read('web_run.json'))['run_id'], run_id)
            for spec in result['receipt']['models']:
                rows = list(csv.DictReader(io.StringIO(z.read(spec['file']).decode())))
                self.assertEqual(len(rows), 400)
                self.assertTrue(all(float(row['prediction']) > 0 for row in rows))
        code, _, _ = self.request('/api/retest/status?run_id=not-current')
        self.assertEqual(code, 400)

    def test_wheel_without_repository_data_still_has_prediction_ui(self):
        with tempfile.TemporaryDirectory() as temp:
            other = make_server(0, temp)
            thread = threading.Thread(target=other.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{other.server_address[1]}'
                with urllib.request.urlopen(url + '/api/info', timeout=30) as r:
                    info = json.load(r)
                self.assertFalse(info['reproduction_available'])
                self.assertFalse(info['retest_available'])
                self.assertFalse(info['training_available'])
                self.assertEqual(len(info['models']), 65)
            finally:
                other.shutdown(); other.server_close(); thread.join(timeout=3)

if __name__ == '__main__':
    unittest.main()
