import sys
from pathlib import Path

import nbformat
from dagster import (AssetKey, Definitions, Nothing, asset, define_asset_job, multiprocess_executor)
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    'preprocessing/mimic': [],
    'preprocessing/eicu': [],
    'features/build': ['preprocessing/mimic', 'preprocessing/eicu'],
    'models/source': ['features/build'],

    'physiology_increment/comparisons': ['models/source'],
    'observation_increment/comparisons': ['models/source'],
    'observation_counts/quantity': ['models/source'],
    'observation_counts/timing': ['models/source'],

    'local_eicu/training': ['models/source'],
    'local_eicu/availability': ['local_eicu/training'],
    'adaptation/local_calibration': ['local_eicu/training'],

    'adaptation/calibration': [
        'observation_counts/quantity',
        'observation_counts/timing'
    ],
    'adaptation/pooled': ['observation_counts/quantity'],

    'posthoc/learner_controls': ['models/source'],
    'posthoc/source_uncertainty': ['models/source'],
    'posthoc/hospital_effects': ['models/source'],
    'posthoc/reciprocal_transport': ['models/source'],
    'posthoc/ventilation': ['models/source'],

    'results/figures': [
        'physiology_increment/comparisons',
        'observation_increment/comparisons',
        'observation_counts/timing',
        'local_eicu/availability',
        'adaptation/local_calibration',
        'posthoc/hospital_effects'
    ],

    'results/tables': [
        'physiology_increment/comparisons',
        'observation_increment/comparisons',
        'observation_counts/timing',
        'local_eicu/availability',
        'posthoc/learner_controls',
        'posthoc/source_uncertainty',
        'posthoc/reciprocal_transport'
    ],

    'results/accompanying': [
        'physiology_increment/comparisons',
        'observation_increment/comparisons',
        'observation_counts/timing',
        'local_eicu/availability',
        'adaptation/local_calibration',
        'adaptation/pooled',
        'posthoc/learner_controls',
        'posthoc/source_uncertainty',
        'posthoc/hospital_effects',
        'posthoc/reciprocal_transport',
        'posthoc/ventilation'
    ]
}


def run_notebook(name, context):
    notebook = nbformat.read(ROOT / 'notebooks' / f'{name}.ipynb', as_version = 4)
    output = ROOT / 'work/executed_notebooks' / f'{name}.ipynb'
    output.parent.mkdir(parents = True, exist_ok = True)

    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    def report(cell, cell_index, **kwargs):
        for item in cell.get('outputs', []):
            if item.output_type == 'stream':
                context.log.info(item.text.rstrip())

    client = NotebookClient(
        notebook, timeout = None, kernel_name = 'python3', resources = {'metadata': {'path': str(ROOT)}}
    )

    client.create_kernel_manager().kernel_spec.argv = [
        sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'
    ]

    client.on_cell_start = lambda cell, cell_index: context.log.info(f'{name}: cell {cell_index + 1}')
    client.on_cell_executed = report

    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)

    context.log.info(f'Executed notebook: {output}')


def notebook_asset(name, dependencies):
    @asset(
        key = AssetKey(name.split('/')),
        deps = [AssetKey(parent.split('/')) for parent in dependencies],
        group_name = name.split('/')[0],
        dagster_type = Nothing
    )
    def execute(context):
        run_notebook(name, context)

    return execute


defs = Definitions(
    assets = [notebook_asset(name, dependencies) for name, dependencies in NOTEBOOKS.items()],
    jobs = [define_asset_job('study')],
    executor = multiprocess_executor
)
