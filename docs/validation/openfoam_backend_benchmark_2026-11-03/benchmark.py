"""Bounded native OpenFOAM I/O and geometry comparison; writes raw JSON results."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from flowtorch.data.foam_dataloader import FOAMDataloader, FOAMCase, FOAMMesh
import flowtorch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('flowtorch.data._foam_baseline', HERE / 'baseline_foam_dataloader.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
logging.disable(logging.CRITICAL)


def hdr(cls, obj, note=''):
    return (f'FoamFile\n{{\nversion 2.0;\nformat binary;\nclass {cls};\n'
            f'arch "LSB;label=32;scalar=64";\nobject {obj};\n{note}\n}}\n').encode()


def block(a, dtype):
    a = np.asarray(a, dtype=dtype)
    return f'{len(a)}\n('.encode() + a.tobytes() + b')\n'


def grid(root, n):
    mesh = root / 'constant/polyMesh'
    mesh.mkdir(parents=True)
    (root / '0').mkdir()
    coords = np.stack(np.meshgrid(*([np.arange(n+1)]*3), indexing='ij'), -1).reshape(-1, 3)
    points = coords.astype(float) / n
    def pid(x):
        return (x[:,0]*(n+1)+x[:,1])*(n+1)+x[:,2]
    def cid(x):
        return (x[:,0]*n+x[:,1])*n+x[:,2]
    internals, boundaries = [], []
    for axis in range(3):
        ranges = [np.arange(n)] * 3
        ranges[axis] = np.arange(n+1)
        c = np.stack(np.meshgrid(*ranges, indexing='ij'), -1).reshape(-1,3)
        a, b = (axis+1)%3, (axis+2)%3
        da, db = np.eye(3, dtype=int)[a], np.eye(3, dtype=int)[b]
        f = np.stack([pid(c), pid(c+da), pid(c+da+db), pid(c+db)], 1)
        low = c[:,axis] == 0
        high = c[:,axis] == n
        f[low] = f[low, ::-1]
        own = c.copy()
        own[:,axis] = np.maximum(c[:,axis]-1, 0)
        mask = ~(low|high)
        internals.append((f[mask], cid(own[mask]), cid(c[mask])))
        boundaries.append((f[~mask], cid(own[~mask])))
    faces = np.concatenate([x[0] for x in internals] + [x[0] for x in boundaries])
    owners = np.concatenate([x[1] for x in internals] + [x[1] for x in boundaries])
    neighbors = np.concatenate([x[2] for x in internals])
    (mesh/'points').write_bytes(hdr('vectorField','points')+block(points,'<f8'))
    (mesh/'faces').write_bytes(hdr('faceCompactList','faces')+block(np.arange(0,4*len(faces)+1,4),'<i4')+block(faces.ravel(),'<i4'))
    (mesh/'owner').write_bytes(hdr('labelList','owner',f'note "nCells:{n**3} nFaces:{len(faces)}";')+block(owners,'<i4'))
    (mesh/'neighbour').write_bytes(hdr('labelList','neighbour')+block(neighbors,'<i4'))
    (mesh/'boundary').write_bytes(hdr('polyBoundaryMesh','boundary')+b'0()')
    return n**3, len(faces)


def time_methods(methods, repeats=5, inner=1):
    # One untimed call per method warms file pages and checks execution.
    for fn in methods.values():
        fn()
    raw = {key: [] for key in methods}
    keys = list(methods)
    for trial in range(repeats):
        order = keys if trial%2 == 0 else keys[::-1]
        for key in order:
            begin = time.perf_counter()
            for _ in range(inner):
                value = methods[key]()
            elapsed = time.perf_counter()-begin
            raw[key].append(elapsed/inner)
            del value
    return {key: {'seconds': samples, 'median_s': statistics.median(samples),
                  'min_s': min(samples), 'max_s': max(samples)} for key,samples in raw.items()}


def fields(root, manufactured=False):
    plain = FOAMDataloader(str(root), dtype=torch.float64)
    baseline = old.FOAMDataloader(str(root), dtype=torch.float64)
    gzroot = root.parent / (root.name+'_gzip')
    # Compress only native case inputs; no generated HDF5 or unrelated files.
    paths = [p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.hdf5','.h5','.xdmf')]
    for path in paths:
        out = gzroot / path.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out)+'.gz').write_bytes(gzip.compress(path.read_bytes(), compresslevel=6, mtime=0))
    zipped = FOAMDataloader(str(gzroot), dtype=torch.float64)
    time_name = '0' if manufactured else '0.5'
    expected = baseline.load_snapshot('U',time_name)
    torch.testing.assert_close(plain.load_snapshot('U',time_name),expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(zipped.load_snapshot('U',time_name),expected,rtol=1e-12,atol=1e-12)
    count = len(expected)
    width = max(1,count//100)
    selected = slice(count//2, count//2+width)
    methods = {
        'old_binary_slice': lambda: baseline.load_snapshot_slice('U',time_name,selected),
        'new_binary_slice': lambda: plain.load_snapshot_slice('U',time_name,selected),
        'new_gzip_slice': lambda: zipped.load_snapshot_slice('U',time_name,selected),
        'old_binary_full': lambda: baseline.load_snapshot('U',time_name),
        'new_binary_full': lambda: plain.load_snapshot('U',time_name),
        'new_gzip_full': lambda: zipped.load_snapshot('U',time_name),
    }
    for key in ('old_binary_slice','new_binary_slice','new_gzip_slice'):
        torch.testing.assert_close(methods[key](),expected[selected],rtol=1e-12,atol=1e-12)
    u = root/time_name/'U'
    uz = gzroot/time_name/'U.gz'
    sizes = {'native_case_binary_bytes': sum(p.stat().st_size for p in paths),
             'native_case_gzip_bytes': sum(p.stat().st_size for p in gzroot.rglob('*.gz')),
             'U_binary_bytes':u.stat().st_size,'U_gzip_bytes':uz.stat().st_size}
    return {'cells':count,'slice_cells':width,'slice_start':selected.start,'sizes':sizes,
            'timings':time_methods(methods,inner=20 if count<1000 else 1)}


def geometry(root, expected_cells, threads):
    torch.set_num_threads(threads)
    current = FOAMMesh(FOAMCase(str(root)),torch.float64)
    previous = old.FOAMMesh(old.FOAMCase(str(root)),torch.float64)
    mesh_path = str(root/'constant/polyMesh')
    new_fn = lambda: current._compute_cell_centers_and_volumes(mesh_path)
    old_fn = lambda: previous._compute_cell_centers_and_volumes(mesh_path)
    a,b = old_fn(),new_fn()
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-12)
    assert len(b[0]) == expected_cells
    if root.name.startswith('grid'):
        torch.testing.assert_close(b[1],torch.full_like(b[1],1/expected_cells),rtol=1e-11,atol=1e-12)
        assert abs(b[1].sum().item()-1)<1e-11
        n = round(expected_cells**(1/3))
        coordinates = torch.arange(n,dtype=torch.float64)
        reference = (torch.stack(torch.meshgrid(coordinates,coordinates,coordinates,indexing='ij'),-1).reshape(-1,3)+0.5)/n
        torch.testing.assert_close(b[0],reference,rtol=1e-11,atol=1e-12)
    timings = time_methods({'old_end_to_end':old_fn,'new_end_to_end':new_fn}, repeats=3)
    # Same already-parsed mesh arrays isolate numerical geometry from file parsing.
    points = current._parse_points(mesh_path)
    counts, faces = current._parse_faces(mesh_path)
    owners, neighbors = current._parse_owners_and_neighbors(mesh_path)
    for obj in (current,previous):
        obj._parse_points = lambda path: points
        obj._parse_faces = lambda path: (counts,faces)
        obj._parse_owners_and_neighbors = lambda path: (owners,neighbors)
        obj._get_n_cells = lambda path: expected_cells
    timings.update(time_methods({'old_compute_only':old_fn,'new_compute_only':new_fn}, repeats=3))
    return {'cells':expected_cells,'faces':len(owners),'threads':threads,'timings':timings}


def main():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    results = {'configuration':{'cpu':open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(' :\t'),
                 'platform':platform.platform(),'torch':torch.__version__,'flowtorch':flowtorch.__file__,
                 'version':flowtorch.__version__,'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'updated_source_sha256':hashlib.sha256(Path(FOAMDataloader.__module__.replace('.','/')+'.py').read_bytes()).hexdigest(),
                 'dtype':'float64','device':'cpu','default_threads':original_threads,'cache':'warm, one warmup per method',
                 'gzip_level':6,'field_repeats':5,'geometry_repeats':3},'fields':{},'geometry':{}}
    out = HERE/'results.json'
    if '--resume' in sys.argv and out.exists():
        results = json.loads(out.read_text())
    def save():
        out.write_text(json.dumps(results,indent=2))
    with tempfile.TemporaryDirectory(prefix='flowtorch-foam-bench-') as tmp:
        tmp = Path(tmp)
        source = Path('datasets/of_cavity_binary').resolve()
        # Avoid copying existing converter artifacts or touching original inputs.
        cavity = tmp/'cavity'
        for folder in ['constant','0','0.1','0.2','0.3','0.4','0.5']:
            for p in (source/folder).rglob('*'):
                if p.is_file():
                    dest = cavity/p.relative_to(source)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(p.read_bytes())
        if 'cavity' not in results['fields']:
            results['fields']['cavity'] = fields(cavity)
        print('Cavity fields complete',flush=True);save()
        big = tmp/'large_field'
        mesh = big/'constant/polyMesh';mesh.mkdir(parents=True)
        for name in ['owner','neighbour','points','faces','boundary']:
            (mesh/name).write_bytes(b'placeholder')
        (big/'0').mkdir()
        n = 250000
        x = np.arange(n,dtype=float)/n
        values = np.stack([np.sin(8*np.pi*x),np.cos(8*np.pi*x),x*x],axis=1)
        # Old parser expects the conventional multiline internalField header.
        data = hdr('volVectorField','U')+b'dimensions [0 1 -1 0 0 0 0];\ninternalField nonuniform List<vector>\n'+block(values,'<f8')+b';\nboundaryField {}\n'
        (big/'0/U').write_bytes(data)
        if 'manufactured_250k' not in results['fields']:
            results['fields']['manufactured_250k'] = fields(big,True)
        print('Large fields complete',flush=True);save()
        if '--ascii' in sys.argv and 'manufactured_250k_ascii' not in results['fields']:
            ascii_root = tmp/'large_ascii_field'
            ascii_mesh = ascii_root/'constant/polyMesh'
            ascii_mesh.mkdir(parents=True)
            for name in ['owner','neighbour','points','faces','boundary']:
                (ascii_mesh/name).write_bytes(b'placeholder')
            (ascii_root/'0').mkdir()
            ascii_data = hdr('volVectorField','U').replace(b'format binary',b'format ascii')
            ascii_data += b'dimensions [0 1 -1 0 0 0 0];\ninternalField nonuniform List<vector>\n'
            ascii_data += f'{n}\n(\n'.encode()
            ascii_data += ''.join('('+' '.join(map(str,row))+')\n' for row in values).encode()
            ascii_data += b');\nboundaryField {}\n'
            (ascii_root/'0/U').write_bytes(ascii_data)
            entry = fields(ascii_root,True)
            entry['timings'] = {key.replace('binary','ascii'): value for key,value in entry['timings'].items()}
            entry['sizes'] = {key.replace('binary','ascii'): value for key,value in entry['sizes'].items()}
            results['fields']['manufactured_250k_ascii'] = entry
            save()
        if '--fields-only' in sys.argv:
            return
        for name, root, count in [('cavity',cavity,400)]:
            if name not in results['geometry']:
                results['geometry'][name] = geometry(root,count,1);save()
        for n in [16,32]:
            root = tmp/f'grid{n}'
            cells,faces = grid(root,n)
            results['geometry'][f'grid{n}_one_thread'] = geometry(root,cells,1)
            print(f'Geometry {n}^3 complete',flush=True);save()
        results['geometry']['grid32_default_threads'] = geometry(root,cells,original_threads)
        save()
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    main()
