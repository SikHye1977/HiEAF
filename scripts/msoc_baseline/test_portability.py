"""Portability must not change ID/label selection or weaken reference verification."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from . import compare
from .common import atomic_json, digest, sha
from .paths import configuration, pgf_metadata_path, ORIGINAL_ROOTS
from src.utils.local_paths import resolve_media_path


class PortabilityTests(unittest.TestCase):
    def test_roots_change_paths_only_and_reject_escape(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            config = root/'data.json'
            atomic_json(config, dict(metadata_root=str(root/'metadata'),
                                    dataset_roots={ds: str(root/ds) for ds in ORIGINAL_ROOTS}))
            c = configuration(config)
            for ds, prefix in ORIGINAL_ROOTS.items():
                result = resolve_media_path(prefix+'/train/person/clip.mp4', c['local_paths'])
                self.assertEqual(result, (root/ds/'train/person/clip.mp4').resolve())
                with self.assertRaises(ValueError):
                    resolve_media_path(prefix+'/../outside.mp4', c['local_paths'])
            entry = {'path': 'D:/datasets/PolyGlotFake/json_file/real_json_file/all_real_video.json'}
            self.assertEqual(pgf_metadata_path(entry, c), root/'PGF/json_file/real_json_file/all_real_video.json')
            packaged = root/'metadata/datasets/PolyGlotFake/json_file/real_json_file/all_real_video.json'
            packaged.parent.mkdir(parents=True)
            packaged.write_text('{}')
            self.assertEqual(pgf_metadata_path(entry, c), packaged)

    def test_reference_works_without_historical_tree_and_rejects_change(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            cohorts = {ds: {'test': [dict(sample_id=ds+':one', clip_label=0),
                                      dict(sample_id=ds+':two', clip_label=1)]} for ds in ORIGINAL_ROOTS}
            reference = dict(format='msoc_hieaf_reference_v1', evidence={}, rows=[{}]*54,
                             cohorts={ds: dict(n=2, sha256=digest(sorted((r['sample_id'], r['clip_label'])
                                              for r in p['test']))) for ds, p in cohorts.items()})
            atomic_json(root/'portable.json', reference)
            atomic_json(root/'reference_portable.lock.json', dict(path='portable.json', sha256=sha(root/'portable.json')))
            with patch.object(compare, 'HERE', root), patch.object(compare, 'configuration', return_value={'metadata_root': root}):
                compare.verify_reference(cohorts)
                bad = copy.deepcopy(cohorts)
                bad['FAV']['test'][0]['clip_label'] = 1
                with self.assertRaisesRegex(ValueError, 'IDs/labels differ'):
                    compare.verify_reference(bad)
                bad = copy.deepcopy(cohorts)
                bad['PGF']['test'][0]['sample_id'] = 'other'
                with self.assertRaisesRegex(ValueError, 'IDs/labels differ'):
                    compare.verify_reference(bad)
                reference['rows'][0] = {'altered': 1}
                atomic_json(root/'portable.json', reference)
                with self.assertRaisesRegex(ValueError, 'reference changed'):
                    compare.verify_reference(cohorts)

    def test_comparison_on_other_machine_keeps_all_54_historical_rows(self):
        import csv
        import io
        with TemporaryDirectory() as t:
            root = Path(t)
            # Deliberately nonexistent old paths: verified export is the transport format.
            rows = []
            cohorts = {}
            output = root/'output'
            for source in ORIGINAL_ROOTS:
                for target in ORIGINAL_ROOTS:
                    pairs = [(target+':real', 0), (target+':fake', 1)]
                    cohorts[target] = dict(n=2, sha256=digest(sorted(pairs)))
                    folder = output/f'source_{source}'/'test'/target
                    folder.mkdir(parents=True)
                    with (folder/'predictions.csv').open('w', newline='') as f:
                        w=csv.writer(f); w.writerow(['sample_id','clip_label']); w.writerows(pairs)
                    atomic_json(folder/'metrics.json', dict(method='MSOC-ResNet', n=2, roc_auc=.6, fake_f1=.5, real_fpr=.5))
                    atomic_json(folder/'COMPLETE.json', dict(metrics_sha256=sha(folder/'metrics.json'),
                                                           predictions_sha256=sha(folder/'predictions.csv')))
                    for condition in ('B','C'):
                        for method in ('baseline','artifact_affine','proposed'):
                            rows.append(dict(source=source,target=target,condition=condition,method=method,
                                             n=2,roc_auc=.8,fake_f1=.7,real_fpr=.3,metrics_path='Z:/absent/metrics.json'))
            ref=dict(format='msoc_hieaf_reference_v1',rows=rows,cohorts=cohorts,evidence={})
            atomic_json(root/'portable.json',ref)
            atomic_json(root/'reference_portable.lock.json',dict(path='portable.json',sha256=sha(root/'portable.json')))
            with patch.object(compare,'HERE',root), patch.object(compare,'configuration',return_value={'metadata_root':root}):
                compare.compare(output)
            result=compare.read_json(output/'COMPARISON_WITH_HIEAF.json')
            self.assertEqual(len(result['rows']),63)
            self.assertEqual(len(result['deltas']),18)
            self.assertAlmostEqual(result['deltas'][0]['roc_auc'],20.)


if __name__ == '__main__':
    unittest.main()
