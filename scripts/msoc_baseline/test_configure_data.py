"""Check portable path discovery and preservation without decoding or training."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from . import configure_data as setup
from .common import bind, read_json, sha


class ConfigureDataTests(unittest.TestCase):
    def roots(self, parent):
        probes = {ds: [Path(ds)/f'{split}.mp4' for split in ('train', 'dev', 'test')]
                  for ds in setup.DATASETS}
        roots = dict(FAV=parent/'FakeAVCeleb_v1.2',
                     AVDF1M=parent/'AVDF1M_MSOC_20261125/data/AV-Deepfake1M',
                     PGF=parent/'PolyGlotFake/PolyGlotFake')
        for ds, folder in roots.items():
            for relative in probes[ds]:
                path = folder/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'root-probe-only')
        return roots, probes

    def test_nested_roots_and_duplicate_need_explicit_selection(self):
        with TemporaryDirectory() as t:
            parent = Path(t).resolve()
            roots, probes = self.roots(parent)
            self.assertEqual(setup.discover_roots(parent, probes), {ds: str(p) for ds, p in roots.items()})
            duplicate = parent/'another_fav'
            for relative in probes['FAV']:
                path = duplicate/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'another-copy')
            with self.assertRaisesRegex(ValueError, 'found 2 possible roots'):
                setup.discover_roots(parent, probes)
            self.assertEqual(setup.discover_roots(parent, probes, {'FAV': roots['FAV']})['FAV'], str(roots['FAV']))

    def test_missing_media_does_not_write_configuration(self):
        with TemporaryDirectory() as t:
            parent = Path(t).resolve()
            roots, probes = self.roots(parent)
            (roots['PGF']/probes['PGF'][0]).write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'PGF: found 0 possible roots'):
                setup.discover_roots(parent, probes)

    def archive(self, parent, name='configs/example.json'):
        archive = parent/'metadata.zip'
        with zipfile.ZipFile(archive, 'x') as z:
            z.writestr(name, '{}')
        archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  metadata.zip\n', encoding='ascii')
        return archive

    def test_extraction_verifies_before_publish_and_preserves_existing(self):
        with TemporaryDirectory() as t:
            parent = Path(t).resolve()
            archive = self.archive(parent)
            destination = parent/'metadata'
            def verify(stage):
                self.assertFalse(destination.exists())
                self.assertEqual((stage/'configs/example.json').read_text(), '{}')
            with patch.object(setup, 'verify_metadata', side_effect=verify) as checker:
                self.assertEqual(setup.unpack_metadata(archive, destination), destination)
                checker.assert_called_once()
                (destination/'keep.txt').write_text('preserve me')
                setup.unpack_metadata(parent/'absent.zip', destination)
                checker.assert_called_once()
            self.assertEqual((destination/'keep.txt').read_text(), 'preserve me')

    def test_checksum_and_unsafe_archive_are_rejected(self):
        with TemporaryDirectory() as t:
            parent = Path(t).resolve()
            archive = self.archive(parent, '../outside.txt')
            destination = parent/'metadata'
            with self.assertRaisesRegex(ValueError, 'Invalid relative path'):
                setup.unpack_metadata(archive, destination)
            self.assertFalse(destination.exists())
            self.assertFalse((parent/'outside.txt').exists())
            archive.with_suffix('.zip.sha256').write_text('0'*64, encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                setup.unpack_metadata(archive, destination)

    def test_failed_verification_never_publishes_metadata(self):
        with TemporaryDirectory() as t:
            parent = Path(t).resolve()
            archive = self.archive(parent)
            destination = parent/'metadata'
            with patch.object(setup, 'verify_metadata', side_effect=ValueError('changed frozen metadata')):
                with self.assertRaisesRegex(ValueError, 'changed frozen metadata'):
                    setup.unpack_metadata(archive, destination)
            self.assertFalse(destination.exists())
            self.assertTrue(archive.exists())

    def test_local_settings_are_idempotent_and_never_replaced(self):
        with TemporaryDirectory() as t:
            path = Path(t)/'data.local.json'
            value = {'metadata_root': t, 'dataset_roots': {'FAV': t}}
            bind(path, value)
            before = path.read_bytes()
            bind(path, value)
            with self.assertRaisesRegex(ValueError, 'Configuration/cohort mismatch'):
                bind(path, {'metadata_root': 'other'})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_json(path), value)


if __name__ == '__main__':
    unittest.main()
