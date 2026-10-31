"""Create a standalone, offline review report without making model calls."""
import argparse
import json
from pathlib import Path


def write_review(results, output_path, recipes_path=None):
    recipes_path = Path(recipes_path) if recipes_path else Path(__file__).with_name('recipes.json')
    payload = json.dumps({'results': results, 'recipes': json.loads(recipes_path.read_text())}, ensure_ascii=False)
    # Embedded model output must never terminate the JSON script element.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template_name = 'comparison_template.html' if any('judge_label' in r for r in results) else 'review_template.html'
    template = Path(__file__).with_name(template_name).read_text(encoding='utf-8')
    path = Path(output_path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(template.replace('__REPORT_DATA__', payload), encoding='utf-8')
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='results.json')
    parser.add_argument('--output', default='results.html')
    args = parser.parse_args()
    write_review(json.loads(Path(args.input).read_text()), args.output)
    print(f'Review report saved to {args.output}')
