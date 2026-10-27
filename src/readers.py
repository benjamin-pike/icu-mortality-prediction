import gzip
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


@contextmanager
def table_source(source, table):
    source = Path(source)
    
    if not source.exists():
        source = source.with_suffix('.zip')
    
    name = table.removesuffix('.csv').lower()
    
    if source.is_dir():
        path = next(
            p
            for p in source.rglob('*')
            if p.is_file()
            and p.name.lower() in {name + '.csv', name + '.csv.gz', name + '.zip'}
        )
        
        if path.suffix == '.zip':
            with ZipFile(path) as archive:
                member = next(n for n in archive.namelist() if Path(n).name.lower() == name + '.csv')
                
                with archive.open(member) as stream:
                    yield stream
        
        elif path.suffix == '.gz':
            with gzip.open(path, 'rb') as stream:
                yield stream
        
        else:
            with path.open('rb') as stream:
                yield stream
    
    else:
        with ZipFile(source) as archive:
            member = next(
                n
                for n in archive.namelist()
                if Path(n).name.lower() in {name + '.csv', name + '.csv.gz'}
            )
            
            with archive.open(member) as stream:
                if member.endswith('.gz'):
                    with gzip.GzipFile(fileobj = stream) as uncompressed:
                        yield uncompressed
                else:
                    yield stream


def read_table(source, table, **kwargs):
    with table_source(source, table) as stream:
        return pd.read_csv(stream, low_memory = False, **kwargs)


def read_chunks(source, table):
    with table_source(source, table) as stream:
        yield from pd.read_csv(stream, chunksize = 100_000, dtype = str, keep_default_na = False)