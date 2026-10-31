import unittest
from pathlib import Path
from calculate_alignment import alignment_metrics
from judge import judge_instructions


class AlignmentTests(unittest.TestCase):
    def test_metrics(self):
        human=[{'question':str(i),'human_label':label} for i,label in enumerate(['bad','good','bad','good'])]
        rows=[dict(row,judge_label=label) for row,label in zip(human,['bad','bad','good','good'])]
        result=alignment_metrics(rows,human)
        self.assertEqual([result[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertEqual([result[k] for k in ('accuracy','precision','recall')],[0.5,0.5,0.5])
        self.assertEqual(alignment_metrics([],[])['precision'],None)
        rows[0]['human_label']='good'
        with self.assertRaises(ValueError):alignment_metrics(rows,human)

    def test_rule_two_is_unchanged(self):
        original=(Path(__file__).resolve().parents[1]/'judge_runs/human-notes-v1/prompt.txt').read_text()
        rule=lambda text:text.split('2. Unsupported explanations')[1].split('3. Ingredient assumptions')[0]
        self.assertEqual(rule(original),rule(judge_instructions))
        self.assertNotIn('Mark bad if ANY',judge_instructions)
        self.assertIn('Distinguish material errors',judge_instructions)
