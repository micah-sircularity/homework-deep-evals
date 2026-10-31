"""Compare saved judge runs against a fixed human-reviewed dataset (no API calls)."""
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def alignment_metrics(rows, human_rows):
    if len(rows) != len(human_rows):
        raise ValueError('Judge and human datasets must have the same number of rows')
    counts = Counter(tp=0, fp=0, fn=0, tn=0)
    mapping = {('bad', 'bad'): 'tp', ('bad', 'good'): 'fp',
               ('good', 'bad'): 'fn', ('good', 'good'): 'tn'}
    for row, reference in zip(rows, human_rows):
        for field, value in reference.items():
            if row.get(field) != value:
                raise ValueError(f'Judge run changed baseline field {field}')
        if reference.get('human_label') not in ('good', 'bad'):
            continue
        try:
            counts[mapping[row['judge_label'], reference['human_label']]] += 1
        except KeyError as error:
            raise ValueError('Judge labels must be good or bad') from error
    tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {**counts, 'total': sum(counts.values()), 'unlabeled': len(rows)-sum(counts.values()),
            'disagreements': fp+fn, 'accuracy': ratio(tp+tn, tp+fp+fn+tn),
            'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn)}


def build_history(root=Path('judge_runs'), human_path=Path('results_reviewed.json')):
    root, human_path = Path(root), Path(human_path)
    human_bytes = human_path.read_bytes()
    human_rows = json.loads(human_bytes)
    runs, datasets = [], []
    for folder in sorted(root.iterdir()):
        source = folder/'results_judged.json'
        if not source.exists():
            continue
        rows = json.loads(source.read_text())
        metrics = alignment_metrics(rows, human_rows)
        prompt = (folder/'prompt.txt').read_bytes()
        runs.append({'rubric_version': folder.name, 'metrics': metrics,
                     'input_sha256': hashlib.sha256(human_bytes).hexdigest(),
                     'prompt_sha256': hashlib.sha256(prompt).hexdigest(),
                     'recipes_sha256': hashlib.sha256((folder/'recipes.json').read_bytes()).hexdigest(),
                     'results_path': str(source), 'html_path': str(source.with_suffix('.html')),
                     'judge_model': rows[0]['judge_model'],
                     'execution_notes': json.loads((folder/'execution_notes.json').read_text()) if (folder/'execution_notes.json').exists() else None,
                     'judge_cost_usd': float(sum((Decimal(str(r['judge_cost'])) for r in rows), Decimal('0'))),
                     'judge_execution_time': sum(r['judge_execution_time'] for r in rows),
                     'judge_tokens': {key: sum(r['judge_tokens'][key] for r in rows)
                                      for key in ('input_tokens','output_tokens','total_tokens','cache_read_tokens','requests')}})
        datasets.append(rows)
        (folder/'metrics.json').write_text(json.dumps(runs[-1], indent=2)+'\n')
    changes = []
    if len(datasets) >= 2:
        for before, after in zip(datasets[-2], datasets[-1]):
            if before['judge_label'] != after['judge_label']:
                changes.append({'question': after['question'], 'human_label': after['human_label'],
                                'human_notes': after.get('human_notes',''),
                                'before': before['judge_label'], 'after': after['judge_label'],
                                'before_reasoning': before['judge_reasoning'],
                                'after_reasoning': after['judge_reasoning']})
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'positive_class': 'bad',
              'human_source': str(human_path), 'runs': runs, 'changed_labels': changes}
    (root/'alignment_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    write_history_html(report, root/'alignment_metrics.html')
    return report


def write_history_html(report, path):
    esc = lambda value: html.escape(str(value))
    percent = lambda value: f'{value*100:.2f}%' if value is not None else 'N/A'
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Judge alignment history</title>
<style>body{font-family:var(--font-sans,system-ui);color:var(--foreground,#253e36);margin:0}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid var(--border,#bdcec6);padding:8px;text-align:left}.scroll{overflow:auto}p,pre{line-height:1.6}pre{white-space:pre-wrap;font:inherit}section{border:1px solid var(--border,#bdcec6);border-radius:var(--radius,8px);padding:16px;margin:16px 0}.barrow{display:grid;grid-template-columns:150px 1fr 65px;gap:8px;align-items:center;margin:8px 0}.track{background:var(--secondary,#e6eeea)}.bar{height:16px;background:var(--chart-1,#248367)}</style>
<p>Positive class: <strong>bad</strong>. All runs use the same saved answers and original human labels. Rule 2 remains unchanged in v2. Costs are estimates for completed judge calls. A single rerun includes model variability; changes cannot be attributed solely to prompt edits.</p><div class="scroll"><table><tr><th>Run</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Disagreements</th><th>Judge USD</th></tr>''']
    for run in report['runs']:
        m=run['metrics']
        cells=[run['rubric_version'],m['tp'],m['fp'],m['fn'],m['tn'],percent(m['accuracy']),percent(m['precision']),percent(m['recall']),m['disagreements'],f"${run['judge_cost_usd']:.8f}"]
        parts.append('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in cells)+'</tr>')
    parts.append('</table></div>')
    if any(run.get('execution_notes') for run in report['runs']):
        parts.append('<p>One v2 call failed at the initial token limit. Its cost and usage are unknown and excluded; the remaining answers were completed by resuming the checkpoint.</p>')
    for metric in ('accuracy','precision','recall'):
        parts.append('<h3>'+metric.title()+'</h3>')
        for index,run in enumerate(report['runs']):
            value=run['metrics'][metric]
            width=100*value if value is not None else 0
            color='var(--chart-1,#248367)' if index==0 else 'var(--chart-2,#6584be)'
            parts.append(f'<div class="barrow"><span>{esc(run["rubric_version"])}</span><div class="track"><div class="bar" style="width:{width:.4f}%;background:{color}"></div></div><span>{percent(value)}</span></div>')
    parts.append(f'<h2>{len(report["changed_labels"])} changed judge labels</h2>')
    for change in report['changed_labels']:
        parts.append(f'<section><h3>{esc(change["question"])}</h3><p>Human: {esc(change["human_label"])} · AI: {esc(change["before"])} → {esc(change["after"])}</p><p>Original human notes: {esc(change["human_notes"] or "None")}</p><details><summary>Previous reasoning</summary><pre>{esc(change["before_reasoning"])}</pre></details><details open><summary>New reasoning</summary><pre>{esc(change["after_reasoning"])}</pre></details></section>')
    parts.append('<details><summary>Provenance, usage, and run metadata</summary><pre>'+esc(json.dumps(report['runs'],indent=2))+'</pre></details></html>')
    Path(path).write_text('\n'.join(parts),encoding='utf-8')


if __name__ == '__main__':
    report=build_history()
    for run in report['runs']:
        print(run['rubric_version'], json.dumps(run['metrics']), 'USD',run['judge_cost_usd'])
    print('Saved judge_runs/alignment_metrics.json and judge_runs/alignment_metrics.html')
