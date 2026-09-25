"""Read-only retrieval. Password is prompted and never saved."""
from pathlib import Path
import getpass,importlib.util,json,re,shlex,sys
LOCAL=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transfer',LOCAL.parent/'scripts/pull_o4b_20260913.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
t.LOCAL=LOCAL
orig=t.paramiko.SFTPFile.prefetch
def bounded(self,file_size=None,max_concurrent_requests=None):
    return orig(self,file_size=file_size,max_concurrent_requests=32)
t.paramiko.SFTPFile.prefetch=bounded
password=getpass.getpass('SSH password (not saved): ')
c=t.connect(password)
try:
    cmd="find /root/autodl-tmp/gw-catalog/packages /root/autodl-tmp/gw-catalog/results /root -maxdepth 4 -name 'gwlr_fpp_injection_real_01_20260922T061000Z_r2_deliverables.tar.gz' -print 2>/dev/null"
    _,out,err=c.exec_command(cmd,timeout=60)
    found=sorted(set(out.read().decode().splitlines()))
    print(json.dumps({'matches':found}),flush=True)
    preferred=[x for x in found if '/gw-catalog/packages/' in x]
    remote=(preferred or found)[0] if found else None
    if not remote:raise RuntimeError('Exact delivery archive not found')
    with c.open_sftp() as s:
        with s.open(remote+'.sha256','rb') as f:side=f.read()
        expected=re.findall(r'\b[0-9a-fA-F]{64}\b',side.decode())[0].lower()
        _,out,err=c.exec_command('sha256sum -- '+shlex.quote(remote),timeout=90)
        actual=out.read().decode().split()[0]
        assert out.channel.recv_exit_status()==0 and actual==expected
        item={'name':Path(remote).name,'remote':remote,'bytes':s.stat(remote).st_size,'remote_sha256':actual,'sidecar_sha256':expected}
        t.save_new(LOCAL/'archives'/(item['name']+'.sha256'),side)
        t.save_new(LOCAL/'audit/REMOTE_DELIVERY.json',json.dumps(item,indent=2).encode())
        print(json.dumps(item),flush=True)
finally:c.close()
r=t.download_retry(password,item)
t.save_new(LOCAL/'audit/DOWNLOAD_COMPLETE.json',json.dumps(r,indent=2).encode())
print('DOWNLOAD VERIFIED',flush=True)
