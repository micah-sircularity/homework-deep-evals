import contextlib
import csv
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic_ai.usage import RunUsage

from run_scenarios import MODEL_NAME, calculate_cost, load_scenarios, run_all


class ScenarioRunnerTests(unittest.TestCase):
    def test_cost_with_cache(self):
        for cached, expected in [(0, '0.65'), (500000, '0.59'), (1000000, '0.53')]:
            with self.subTest(cached=cached):
                usage = RunUsage(input_tokens=1000000, output_tokens=1000000, cache_read_tokens=cached)
                self.assertEqual(calculate_cost(usage), Decimal(expected))
        self.assertEqual(calculate_cost(RunUsage()), Decimal('0'))
        self.assertEqual(calculate_cost(RunUsage(input_tokens=1)), Decimal('0.00000015'))

    def test_reports_aggregate_usage_and_preserves_judge_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            scenarios, output = self.make_paths(directory)
            # Simulate both model responses of a tool-using agent run.
            usage = RunUsage()
            usage.incr(RunUsage(input_tokens=100, output_tokens=20, requests=1))
            usage.incr(RunUsage(input_tokens=200, output_tokens=30, cache_read_tokens=80, requests=1))
            agent = self.make_agent()
            agent.run_sync.return_value = SimpleNamespace(output='Full answer\nSecond line', usage=lambda: usage)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(run_all(scenarios, output, agent), 0)
            rows = json.loads(output.read_text())
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['tokens'], dict(input_tokens=300, output_tokens=50,
                                                     total_tokens=350, cache_read_tokens=80, requests=2))
                self.assertAlmostEqual(row['cost'], 0.0000604)
                self.assertEqual(row['output'], 'Full answer\nSecond line')
                self.assertTrue({'question', 'category', 'type', 'execution_time', 'model'} <= row.keys())
                self.assertEqual(row['pricing']['currency'], 'USD')
            text = stdout.getvalue()
            self.assertNotIn('Full answer\nSecond line', text)
            report = output.with_suffix('.html').read_text()
            self.assertIn('Full answer', report)
            self.assertIn('Export reviewed JSON', report)
            self.assertIn('model requests=4', text)
            self.assertIn('$0.0001208000', text)
            self.assertEqual(agent.run_sync.call_count, 2)

    def test_failure_preserves_completed_results_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            scenarios, output = self.make_paths(directory)
            output.write_text('[{"old": true}]')
            agent = self.make_agent()
            usage = RunUsage(input_tokens=10, output_tokens=5, requests=1)

            def respond(question):
                if question == 'Second?':
                    self.assertEqual(len(json.loads(output.read_text())), 1)
                    raise RuntimeError('API unavailable')
                return SimpleNamespace(output='First answer', usage=lambda: usage)

            agent.run_sync.side_effect = respond
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(run_all(scenarios, output, agent), 1)
            rows = json.loads(output.read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['output'], 'First answer')
            self.assertIn('Failed question: Second?', stderr.getvalue())
            self.assertIn('unknown and excluded', stderr.getvalue())

    def test_invalid_csv_before_api_call_or_result_replacement(self):
        for content in ('question,category\nHello,test\n',
                        'question,category,type\nHello,test,direct\nBad,,direct\n',
                        'question,category,type\nHello,test,direct,extra\n'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                scenarios, output = self.make_paths(directory)
                scenarios.write_text(content)
                output.write_text('[]')
                agent = self.make_agent()
                with self.assertRaises(ValueError):
                    run_all(scenarios, output, agent)
                agent.run_sync.assert_not_called()
                self.assertEqual(output.read_text(), '[]')

    def test_dataset_has_21_scenarios_and_almond_milk_question(self):
        rows = load_scenarios(Path(__file__).resolve().parents[1] / 'scenarios.csv')
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[-1], dict(question='Can I substitute almond milk in the pancakes?',
                                       category='unsupported-advice', type='substitution'))

    def test_report_escapes_untrusted_content(self):
        from review_results import write_review
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.html'
            write_review([{'question': '</script><script>alert(1)</script>',
                           'output': '<img src=x onerror=alert(1)>'}], path)
            html = path.read_text()
            self.assertNotIn('</script><script>alert(1)</script>', html)
            payload = html.split('<script type="application/json" id="data">')[1].split('</script>')[0]
            self.assertEqual(json.loads(payload)['results'][0]['output'], '<img src=x onerror=alert(1)>')

    @staticmethod
    def make_agent():
        return SimpleNamespace(model=SimpleNamespace(model_name=MODEL_NAME), run_sync=Mock())

    @staticmethod
    def make_paths(directory):
        scenarios = Path(directory) / 'scenarios.csv'
        with scenarios.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([['question', 'category', 'type'], ['First?', 'test', 'direct'],
                              ['Second?', 'test', 'direct']])
        return scenarios, Path(directory) / 'results.json'


if __name__ == '__main__':
    unittest.main()
