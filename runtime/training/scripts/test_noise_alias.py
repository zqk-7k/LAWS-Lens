import tempfile
import unittest
from pathlib import Path

from noise_alias import ensure_noise_alias


class NoiseAliasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw.npy'
        self.raw.write_bytes(b'frozen-noise')
        self.target = self.root/'plan.npy'
        self.target.symlink_to(self.raw)
        self.link = self.root/'alias.npy'

    def test_create_and_repeat_nested_alias(self):
        ensure_noise_alias(self.link, self.target)
        ensure_noise_alias(self.link, self.target)
        self.assertTrue(self.link.samefile(self.raw))
        self.assertNotEqual(self.link.resolve(), self.target)

    def test_reject_other_run_even_if_bytes_identical(self):
        other = self.root/'other.npy'
        other.write_bytes(self.raw.read_bytes())
        self.link.symlink_to(other)
        with self.assertRaises(RuntimeError):
            ensure_noise_alias(self.link, self.target)
        self.assertEqual(self.link.resolve(), other)

    def test_reject_dangling_alias_without_repointing(self):
        self.link.symlink_to(self.root/'missing.npy')
        with self.assertRaises(RuntimeError):
            ensure_noise_alias(self.link, self.target)
        self.assertEqual(self.link.readlink(), self.root/'missing.npy')

    def test_reject_missing_target(self):
        with self.assertRaises(FileNotFoundError):
            ensure_noise_alias(self.link, self.root/'missing.npy')
        self.assertFalse(self.link.is_symlink())

    def test_accept_direct_reference_to_same_file(self):
        self.link.symlink_to(self.raw)
        ensure_noise_alias(self.link, self.target)


if __name__ == '__main__':
    unittest.main()
