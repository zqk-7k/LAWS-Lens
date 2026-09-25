"""Read-only retrieval; password is never stored."""
import getpass
import importlib.util
import json
from pathlib import Path
import re
import shlex

LOCAL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transfer', LOCAL.parent / 'scripts/pull_o4b_20260913.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
t.LOCAL = LOCAL
remote = '/root/autodl-tmp/gw-catalog/results/gwlr_tail02_20260922T113500Z_r2_diagnostic_deliverables_v2.tar.gz'
password = getpass.getpass('SSH password (not saved): ')
c = t.connect(password)
try:
    with c.open_sftp() as sftp:
        size = sftp.stat(remote).st_size
        with sftp.open(remote + '.sha256', 'rb') as f:
            side = f.read(65536)
    hashes = re.findall(rb'\b[0-9a-fA-F]{64}\b', side)
    assert len(hashes) == 1
    expected = hashes[0].decode().lower()
    _, out, err = c.exec_command('sha256sum -- ' + shlex.quote(remote), timeout=120)
    actual = out.read().decode().split()[0]
    assert out.channel.recv_exit_status() == 0, err.read().decode()
    assert actual == expected
    item = dict(name=remote.rsplit('/', 1)[1], remote=remote, bytes=size, remote_sha256=actual)
    t.save_new(LOCAL/'archives'/(item['name']+'.sha256'), side)
    t.save_new(LOCAL/'audit/REMOTE_DELIVERY.json', json.dumps(item, indent=2).encode())
    print(json.dumps(item), flush=True)
finally:
    c.close()
t.download(password, item)
