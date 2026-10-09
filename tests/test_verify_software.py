import hashlib
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('verify', Path(__file__).resolve().parents[1]/'tools/verify_software.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

class IntegrityTests(unittest.TestCase):
    def row(self, data):
        return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

    def test_exact(self):
        self.assertTrue(verify.matches(b'a\nb\n', self.row(b'a\nb\n')))

    def test_crlf_checkout(self):
        self.assertTrue(verify.matches(b'a\r\nb\r\n', self.row(b'a\nb\n')))

    def test_crlf_original(self):
        self.assertTrue(verify.matches(b'a\nb\n', self.row(b'a\r\nb\r\n')))

    def test_corruption(self):
        self.assertFalse(verify.matches(b'a\nc\n', self.row(b'a\nb\n')))

    def test_size(self):
        row = self.row(b'a\n'); row['bytes'] += 1
        self.assertFalse(verify.matches(b'a\n',row))

    def test_binary_unchanged(self):
        data=b'\x00\r\n'
        self.assertTrue(verify.matches(data,self.row(data)))

    def test_binary_no_normalization(self):
        self.assertFalse(verify.matches(b'\x00\r\n',self.row(b'\x00\n')))

if __name__ == '__main__':
    unittest.main()
