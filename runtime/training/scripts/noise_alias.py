"""Validate the file identity through nested, frozen noise-bank symlinks."""
from pathlib import Path


def ensure_noise_alias(link, target):
    link, target = Path(link), Path(target)
    target.resolve(strict=True)
    if link.exists() or link.is_symlink():
        if not link.exists() or not link.samefile(target):
            raise RuntimeError('Noise alias points to another deployment: '+str(link))
    else:
        link.symlink_to(target)
    if not link.samefile(target):
        raise RuntimeError('Noise alias identity verification failed')
