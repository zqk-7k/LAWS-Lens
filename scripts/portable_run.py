"""Relocate legacy code imports, refusing reads from the historical project."""
import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('script', type=Path)
    ap.add_argument('args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    root = a.release.resolve()
    old = Path('/root/autodl-tmp/gw-catalog')
    new = root/'runtime/project'
    environment_roots = tuple({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
    def in_environment(path):
        return any(path.is_relative_to(prefix) for prefix in environment_roots)
    redirects, violations = [], []
    def relocate(value):
        if not isinstance(value, (str, Path)):
            return value
        p = Path(value)
        if p.is_absolute() and p.is_relative_to(old) and not p.is_relative_to(root) and not in_environment(p):
            target = new/p.relative_to(old)
            redirects.append(dict(old=str(p), new=str(target)))
            return str(target)
        return str(value)
    class PortablePath(list):
        def insert(self, index, value):
            super().insert(index, relocate(value))
        def append(self, value):
            super().append(relocate(value))
        def extend(self, values):
            super().extend(relocate(x) for x in values)
        def __setitem__(self, key, value):
            super().__setitem__(key, [relocate(x) for x in value] if isinstance(key, slice) else relocate(value))
    sys.path = PortablePath([relocate(x) for x in sys.path])
    original = importlib.util.spec_from_file_location
    def spec(name, location=None, *args, **kwargs):
        return original(name, relocate(location) if location is not None else None, *args, **kwargs)
    importlib.util.spec_from_file_location = spec
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
            if p.is_absolute() and p.is_relative_to(old) and not p.is_relative_to(root) and not in_environment(p):
                violations.append(str(p))
                raise PermissionError('Historical project access blocked by release guard: '+str(p))
    sys.addaudithook(guard)
    sys.argv = [str(a.script), *a.args]
    status = 'PASS'
    try:
        runpy.run_path(str(a.script), run_name='__main__')
    except BaseException:
        status = 'FAIL'
        raise
    finally:
        loaded = sorted(set(str(Path(m.__file__).resolve()) for m in sys.modules.values()
            if getattr(m, '__file__', None) and str(old) in str(m.__file__)))
        report = root/'verification'/('PORTABILITY_'+a.script.stem+'.json')
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(dict(status=status, violations=violations,
            redirected_imports=redirects, loaded_project_modules=loaded,
            permitted_environment_roots=[str(x) for x in environment_roots],
            historical_project_reads_blocked=True, fresh_machine_test=False), indent=2))

if __name__ == '__main__':
    main()
