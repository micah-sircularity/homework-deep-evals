import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from pydantic_ai.usage import RunUsage
from judge import MODEL_NAME, JudgeEvaluation, build_prompt, comparison, judge_instructions, run_judge


class JudgeTests(unittest.TestCase):
    def test_blind_prompt_and_rubric(self):
        prompt = json.loads(build_prompt({'question': 'Q', 'output': 'A', 'human_label': 'SECRET',
                                         'human_notes': 'PRIVATE'}, [{'name': 'Ground truth'}]))
        self.assertEqual(set(prompt), {'question', 'agent_response', 'recipe_collection'})
        self.assertNotIn('SECRET', json.dumps(prompt))
        self.assertNotIn('PRIVATE', json.dumps(prompt))
        for term in ('substitutions', 'Ingredient assumptions', 'Missing recipes', 'restrictions',
                     'Unsupported explanations', 'nutrition', 'Out of scope'):
            self.assertIn(term, judge_instructions)

    def test_comparison(self):
        for human, ai, expected in [('good','bad','disagree'), ('bad','good','disagree'),
                                     ('good','good','agree'), ('bad','bad','agree'), ('','bad','unlabeled')]:
            self.assertEqual(comparison(human, ai), expected)

    def test_preserves_baseline_cost_and_partial_save(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'input.json', Path(directory)/'output.json'
            rows = [{'question': 'Q1', 'output': 'A1', 'human_label': 'good', 'human_notes': 'Original',
                     'tokens': {'input_tokens': 999}, 'cost': 42},
                    {'question': 'Q2', 'output': 'A2', 'human_label': 'bad'}]
            source.write_text(json.dumps(rows))
            evaluator = SimpleNamespace(model=SimpleNamespace(model_name=MODEL_NAME),run_sync=Mock())
            result = SimpleNamespace(output=JudgeEvaluation(label='bad', reasoning='Unsupported advice'),
                                     usage=RunUsage(input_tokens=100,output_tokens=20,requests=1))
            def call(prompt, **kwargs):
                if json.loads(prompt)['question'] == 'Q2':
                    self.assertEqual(len(json.loads(target.read_text())), 1)
                    raise RuntimeError('offline failure')
                return result
            evaluator.run_sync.side_effect=call
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_judge(source,target,evaluator),1)
            row=json.loads(target.read_text())[0]
            for k,v in rows[0].items(): self.assertEqual(row[k],v)
            self.assertEqual(row['agreement'],'disagree')
            self.assertAlmostEqual(row['judge_cost'],0.000025)
            self.assertEqual(json.loads(source.read_text()),rows)
            html=target.with_suffix('.html').read_text()
            self.assertIn('Your original review',html)
            self.assertIn('Follow-up decision',html)
            self.assertIn('You good / AI bad',html)

    def test_successful_run_all_comparison_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'input.json', Path(directory)/'output.json'
            rows=[{'question':str(i), 'output':'Saved answer', 'human_label':label,
                   'human_notes':'</script><script>unsafe()</script>'}
                  for i,label in enumerate(['good','bad','good',''])]
            source.write_text(json.dumps(rows))
            evaluator=SimpleNamespace(model=SimpleNamespace(model_name=MODEL_NAME),run_sync=Mock())
            evaluator.run_sync.side_effect=[SimpleNamespace(
                output=JudgeEvaluation(label=label,reasoning='Recipe evidence'),
                usage=RunUsage(input_tokens=1,output_tokens=1,requests=1))
                for label in ['bad','good','good','bad']]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_judge(source,target,evaluator),0)
            saved=json.loads(target.read_text())
            self.assertEqual([r['agreement'] for r in saved],['disagree','disagree','agree','unlabeled'])
            html=target.with_suffix('.html').read_text()
            self.assertNotIn('</script><script>unsafe()</script>',html)
            for call in evaluator.run_sync.call_args_list:
                self.assertNotIn('human_notes',call.args[0])
                self.assertNotIn('human_label',call.args[0])

    def test_resume_skips_completed_and_rejects_changed_baseline(self):
        from judge import RUBRIC_VERSION
        with tempfile.TemporaryDirectory() as directory:
            source,target=Path(directory)/'input.json',Path(directory)/'output.json'
            rows=[{'question':'Q1','output':'A1','human_label':'good'},
                  {'question':'Q2','output':'A2','human_label':'bad'}]
            source.write_text(json.dumps(rows))
            first=dict(rows[0],judge_label='good',judge_reasoning='Grounded',agreement='agree',
                       judge_rubric_version=RUBRIC_VERSION,judge_model=MODEL_NAME,
                       judge_tokens={},judge_cost=0)
            target.write_text(json.dumps([first]))
            evaluator=SimpleNamespace(model=SimpleNamespace(model_name=MODEL_NAME),run_sync=Mock())
            evaluator.run_sync.return_value=SimpleNamespace(
                output=JudgeEvaluation(label='bad',reasoning='Unsupported'),usage=RunUsage())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_judge(source,target,evaluator,resume=True),0)
            evaluator.run_sync.assert_called_once()
            self.assertEqual(json.loads(evaluator.run_sync.call_args.args[0])['question'],'Q2')
            self.assertEqual(json.loads(target.read_text())[0],first)
            rows[0]['human_label']='bad';source.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):run_judge(source,target,evaluator,resume=True)

    def test_invalid_input_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'bad.json';source.write_text('[{"question":"Q"}]')
            evaluator=Mock()
            with self.assertRaises(ValueError):run_judge(source,Path(directory)/'out.json',evaluator)
            evaluator.run_sync.assert_not_called()
