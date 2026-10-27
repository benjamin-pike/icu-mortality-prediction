import os
import pickle
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('ICU_RAW_DATA', ROOT.parent / 'data'))
WORK = ROOT / 'work'
RESOURCES = ROOT / 'resources'


def reset_directory(path):
    if path.exists():
        shutil.rmtree(path)
    
    path.mkdir(parents = True)


def save(path, value):
    path.parent.mkdir(parents = True, exist_ok = True)
    
    with path.open('wb') as stream:
        pickle.dump(value, stream, protocol = pickle.HIGHEST_PROTOCOL)


def load(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def write_table(path, frame):
    path.parent.mkdir(parents = True, exist_ok = True)
    frame = frame.copy()
    
    for name in (
        'domain',
        'source',
        'support',
        'tier',
        'partition',
        'restriction',
        'training',
        'calibration',
        'method',
        'learner',
        'condition',
        'group'
    ):
        if name in frame:
            frame[name] = frame[name].astype('category')
    
    frame.to_parquet(path)
