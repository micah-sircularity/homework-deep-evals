import csv
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

from review_results import write_review

MODEL_NAME = 'accounts/fireworks/models/glm-5p3-flash'
PRICING = {
    'currency': 'USD',
    'estimated': True,
    'input_per_million': 0.15,
    'cached_input_per_million': 0.03,
    'output_per_million': 0.50,
    'source': 'https://fireworks.ai/models/fireworks/glm-5p3-flash',
    'verified_date': '2026-10-31',
}


def calculate_cost(usage):
    """Estimate USD cost from aggregate run usage, including cached input."""
    return (
        Decimal(usage.input_tokens - usage.cache_read_tokens) * Decimal('0.15')
        + Decimal(usage.cache_read_tokens) * Decimal('0.03')
        + Decimal(usage.output_tokens) * Decimal('0.50')
    ) / Decimal('1000000')


def load_scenarios(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ('question', 'category', 'type')
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError('CSV must have question, category, and type columns')
        rows = list(reader)
        for number, row in enumerate(rows, start=2):
            if None in row or any(not (row.get(key) or '').strip() for key in required):
                raise ValueError(f'CSV row {number} must have nonempty question, category, and type fields and no extra values')
        return rows


def save_results(path, results):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    write_review(results, path.with_suffix(".html"))


def print_summary(results, elapsed, output_path):
    keys = ('input_tokens', 'output_tokens', 'total_tokens', 'cache_read_tokens', 'requests')
    totals = {key: sum(row['tokens'][key] for row in results) for key in keys}
    cost = sum((Decimal(str(row['cost'])) for row in results), Decimal('0'))
    print(f'\nCompleted {len(results)} scenarios in {elapsed:.2f}s; saved to {output_path}')
    print(f"Tokens: input={totals['input_tokens']}, output={totals['output_tokens']}, "
          f"total={totals['total_tokens']}, cached input={totals['cache_read_tokens']}; "
          f"model requests={totals['requests']}")
    print(f'Total estimated cost (completed scenarios): ${cost:.10f} USD (not invoice charges)')


def run_all(scenarios_path='scenarios.csv', output_path='results.json', recipe_agent=None):
    scenarios = load_scenarios(scenarios_path)
    if recipe_agent is None:
        from recipe_agent import agent
        recipe_agent = agent
    if recipe_agent.model.model_name != MODEL_NAME:
        raise ValueError(f'Pricing is only configured for {MODEL_NAME}')

    results = []
    start = time.perf_counter()
    save_results(output_path, results)
    try:
        for i, scenario in enumerate(scenarios, start=1):
            question = scenario['question']
            print(f'\n[{i}/{len(scenarios)}] {question}', flush=True)
            scenario_start = time.perf_counter()
            try:
                result = recipe_agent.run_sync(question)
            except Exception as error:
                print(f'Failed question: {question}\n{type(error).__name__}: {error}', file=sys.stderr)
                print('Failed request usage/cost is unknown and excluded from totals.', file=sys.stderr)
                return 1
            elapsed = time.perf_counter() - scenario_start
            usage = result.usage()
            cost = calculate_cost(usage)
            results.append({
                'question': question,
                'category': scenario['category'],
                'type': scenario['type'],
                'output': result.output,
                'execution_time': round(elapsed, 4),
                'tokens': {
                    'input_tokens': usage.input_tokens,
                    'output_tokens': usage.output_tokens,
                    'total_tokens': usage.total_tokens,
                    'cache_read_tokens': usage.cache_read_tokens,
                    'requests': usage.requests,
                },
                'model': recipe_agent.model.model_name,
                'cost': float(cost),
                'pricing': dict(PRICING),
            })
            save_results(output_path, results)
            print(f'Saved scenario {i}/{len(scenarios)} to {output_path} and its HTML review report.', flush=True)
    finally:
        print_summary(results, time.perf_counter() - start, output_path)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run_all())
    except (ValueError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
