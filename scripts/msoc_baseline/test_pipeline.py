"""Meaningful adapter, cache, gradient and checkpoint-resume regression checks."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from types import MethodType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from . import model, train, cache
from .common import atomic_json, bind, digest, Stopped
from .media import pack, unpack
from .data import epoch_order


class TinyMSOC(nn.Module):
    def __init__(self, *_):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.drop = nn.Dropout(.2)
        self.fc = nn.Linear(3, 2)

    def forward(self, video, audio):
        return self.fc(self.drop(self.bn(video.mean((2, 3, 4)))))

    def objective(self, output, labels):
        return nn.functional.cross_entropy(output, labels[:, 0])

    def scores(self, output):
        p = output.softmax(1)[:, 1:2]
        return torch.cat((p, p, p, 1-p), dim=1)


class NoStop:
    def check(self):
        pass

    def pending(self):
        return False


class StopAfterUpdate(NoStop):
    def pending(self):
        return True

    def check(self):
        raise Stopped('test interruption')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_official_oc_and_score_polarity(self):
        torch.manual_seed(4)
        ours = model.OCSoftmax(8)
        official = model.upstream('util/loss.py').OCSoftmax(feat_dim=8)
        official.load_state_dict(ours.state_dict())
        x = torch.randn(6, 8)
        real = torch.tensor([0, 1, 0, 1, 0, 1])
        expected, score = official(x, real)
        torch.testing.assert_close(ours(x, real), expected)
        torch.testing.assert_close(ours.score(x), score)
        with self.assertRaises(ValueError):
            ours(x, torch.tensor([-1]*6))
        m = train.metrics([0, 0, 1, 1], [.1, .6, .5, .9])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(m['real_fpr'], .5)
        self.assertEqual(m['fake_f1'], .8)

    def test_frontend_gradient_and_scnet_frame_order(self):
        encoder = model.ResEncoder('relu', light=True)
        x = torch.randn(2, 3, 3, 32, 32)
        encoder(x).square().mean().backward()
        self.assertGreater(float(encoder.frontend3D[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(encoder.frontend3D[1].weight.grad.abs().sum()), 0)
        # Test exact SCNet input ordering independently of the expensive network.
        m = model.MSOC.__new__(model.MSOC)
        nn.Module.__init__(m)
        m.backbone = 'scnet'
        class Record(nn.Module):
            def forward(self, x):
                self.seen = x.detach().clone()
                return x.mean((2, 3))
        m.v_visualFrontend = Record()
        x = torch.arange(2*3*30*2*2).reshape(2, 3, 30, 2, 2).float()
        m.visual(x)
        torch.testing.assert_close(m.v_visualFrontend.seen[31], x[1, :, 1])

    def test_full_adapter_matches_official_composition_loss_and_scores(self):
        # Execute the actual upstream forward/loss methods without importing Lightning
        # or its global 'models' package. Use the same corrected frontend on both sides.
        from .common import HERE
        tree = ast.parse((HERE/'upstream/src/models/talkNet/MSOC.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MSOC')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and (n.name.startswith('forward') or n.name == 'loss_fn')]
        scope = {'torch': torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<upstream-forward-loss>', 'exec'), scope)
        torch.manual_seed(4)
        ours = model.MSOC().eval()
        official = copy.deepcopy(ours).eval()
        official.scnet = False
        for node in methods:
            setattr(official, node.name, MethodType(scope[node.name], official))
        OC = model.upstream('util/loss.py').OCSoftmax
        for name in ('loss_audio', 'loss_video', 'av_loss_audio', 'av_loss_video'):
            layer = OC(feat_dim=128)
            layer.load_state_dict(getattr(ours, name).state_dict())
            setattr(official, name, layer)
        official.mm_cls, official.softmax = nn.CrossEntropyLoss(), nn.Softmax(dim=1)
        official.save_score, official.middle_infer = False, False
        official.visual_threshold = official.audio_threshold = official.final_threshold = .5
        official.pred_strategy = 'mean'
        v, a = torch.randn(2, 3, 30, 32, 32), torch.randn(2, 120, 13)
        labels = torch.tensor([[0, 0, 0], [1, 1, 0]])
        with torch.no_grad():
            one, two = ours(v, a), official(v, a)
            for left, right in zip(one, two):
                torch.testing.assert_close(left, right)
            result = official.loss_fn(*two, 1-labels[:, 1], 1-labels[:, 2], 1-labels[:, 0])
            torch.testing.assert_close(ours.objective(one, labels), result['loss_dict']['loss'])
            np.testing.assert_allclose(ours.scores(one)[:, 0].numpy(), 1-result['scores'])

    def test_cache_integrity_and_reproducible_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'inputs.sqlite'
            rows = [dict(sample_id=str(i), clip_label=i%2, video_label=i%2, audio_label=i%2) for i in range(6)]
            self.make_cache(path, rows)
            first = cache.CachedDataset(path, rows, 'train', epoch_order(rows, 42, 3), epoch=3)
            later = cache.CachedDataset(path, rows, 'train', epoch_order(rows, 42, 3), epoch=3, start=2)
            torch.testing.assert_close(first[2][0], later[0][0])
            self.assertEqual(first[2][3], later[0][3])
            first.close()
            later.close()
            db = cache.db_open(path)
            db.execute("UPDATE inputs SET sha='bad' WHERE sid='0'")
            db.commit()
            db.close()
            corrupt = cache.CachedDataset(path, rows, 'train')
            with self.assertRaisesRegex(ValueError, 'Corrupt cached'):
                corrupt[0]
            corrupt.close()
            bind(Path(tmp)/'id.json', {'id': 1})
            with self.assertRaises(ValueError):
                bind(Path(tmp)/'id.json', {'id': 2})

    def make_cache(self, path, rows, train=True):
        db = cache.db_open(path)
        db.execute('CREATE TABLE inputs(sid TEXT PRIMARY KEY,blob BLOB,sha TEXT,info TEXT)')
        side = 128 if train else 100
        for i, row in enumerate(rows):
            rng = np.random.default_rng(i)
            v = rng.integers(0, 256, (30, side, side, 3), dtype=np.uint8)
            a = rng.normal(size=(120, 13)).astype('float32')
            blob = pack(v, a)
            vv, aa = unpack(blob, side)
            np.testing.assert_array_equal(vv, v)
            np.testing.assert_array_equal(aa, a)
            db.execute('INSERT INTO inputs VALUES (?,?,?,?)', (row['sample_id'], blob, hashlib.sha256(blob).hexdigest(), '{}'))
        db.commit()
        db.close()

    def test_resume_preserves_optimizer_rng_bn_and_last_partial_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = {s: [dict(sample_id=f'{s}-{i}', clip_label=i%2, video_label=i%2, audio_label=i%2) for i in range(n)]
                    for s, n in [('train', 6), ('dev', 4)]}
            paths = {s: root/f'{s}.sqlite' for s in rows}
            for s in rows:
                self.make_cache(paths[s], rows[s], train=s == 'train')
            args = SimpleNamespace(seed=42, device='cpu', precision='fp32', cpu_threads=2, backbone='resnet',
                                   batch_size=2, accumulation=2, workers=0, eval_batch=2, save_seconds=180, epochs=2)
            identity = {'test': 'resume'}
            with patch.object(train, 'MSOC', TinyMSOC):
                train.fit(root/'full', paths, rows, args, identity, NoStop())
                with self.assertRaises(Stopped):
                    train.fit(root/'resume', paths, rows, args, identity, StopAfterUpdate())
                partial = train.load_checkpoint(root/'resume/latest.pt')
                self.assertEqual(partial['state']['cursor'], 4)
                train.fit(root/'resume', paths, rows, args, identity, NoStop())
            a, b = (train.load_checkpoint(root/k/'latest.pt') for k in ('full', 'resume'))
            for k in a['model']:
                torch.testing.assert_close(a['model'][k], b['model'][k], rtol=0, atol=0)
            for i in a['optimizer']['state']:
                for k in a['optimizer']['state'][i]:
                    torch.testing.assert_close(a['optimizer']['state'][i][k], b['optimizer']['state'][i][k], rtol=0, atol=0)
            torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'], rtol=0, atol=0)
            self.assertEqual(a['state']['updates'], 4)
            self.assertEqual(a['state']['best_epoch'], b['state']['best_epoch'])

    def test_prediction_resume_unknown_modality_coverage_and_lock(self):
        from . import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(sample_id=str(i), group_id=f'g{i}', clip_label=i%2, video_label=None,
                         audio_label=None) for i in range(4)]
            self.make_cache(root/'test.sqlite', rows, train=False)
            args = SimpleNamespace(device='cpu', cpu_threads=2, backbone='resnet', workers=0, eval_batch=2)
            identity = {'test': 'prediction resume'}
            torch.manual_seed(15)
            tiny = TinyMSOC()
            train.save_checkpoint(root/'best.pt', dict(identity={'run': identity}, model=tiny.state_dict()))
            class StopSecond(NoStop):
                count = 0
                def check(self):
                    self.count += 1
                    if self.count == 2:
                        raise Stopped('after first prediction commit')
            with patch.object(evaluate, 'MSOC', TinyMSOC):
                with self.assertRaises(Stopped):
                    evaluate.evaluate(root/'partial', root/'best.pt', root/'test.sqlite', rows, args, identity, StopSecond())
                db = cache.db_open(root/'partial/scores.sqlite', readonly=True)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM scores').fetchone()[0], 2)
                db.close()
                a = evaluate.evaluate(root/'partial', root/'best.pt', root/'test.sqlite', rows, args, identity, NoStop())
                b = evaluate.evaluate(root/'full', root/'best.pt', root/'test.sqlite', rows, args, identity, NoStop())
                self.assertEqual(a, b)
                self.assertEqual(a['n'], 4)
                self.assertEqual(a['unknown_modality_labels'], 4)
                self.assertEqual((root/'partial/predictions.csv').read_bytes(), (root/'full/predictions.csv').read_bytes())
                with self.assertRaises(ValueError):
                    evaluate.evaluate(root/'partial', root/'best.pt', root/'test.sqlite', rows, args, {'wrong': 1}, NoStop())

    def test_preparation_commits_each_sample_and_resumes_only_missing(self):
        from .data import stamp
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root/'dummy.media'
            media.write_bytes(b'cache-unit-fixture')
            rows = [dict(sample_id=str(i), path=str(media), stamp=stamp(media), clip_label=i%2,
                         video_label=i%2, audio_label=i%2) for i in range(3)]
            decoded = (np.zeros((30, 100, 100, 3), dtype=np.uint8), np.zeros((120, 13), dtype=np.float32), {})
            class StopSecond(NoStop):
                count = 0
                def check(self):
                    self.count += 1
                    if self.count == 2:
                        raise Stopped('after one commit')
            with patch.object(cache, 'decode', return_value=decoded):
                with self.assertRaises(Stopped):
                    cache.prepare(root/'inputs.sqlite', rows, 'test', 1, StopSecond())
            db = cache.db_open(root/'inputs.sqlite', readonly=True)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM inputs').fetchone()[0], 1)
            saved = db.execute('SELECT blob FROM inputs').fetchone()[0]
            db.close()
            with patch.object(cache, 'decode', return_value=decoded) as decode:
                cache.prepare(root/'inputs.sqlite', rows, 'test', 1, NoStop())
                self.assertEqual(decode.call_count, 2)
            db = cache.db_open(root/'inputs.sqlite', readonly=True)
            self.assertEqual(db.execute('SELECT blob FROM inputs WHERE sid=\'0\'').fetchone()[0], saved)
            db.close()
            changed = copy.deepcopy(rows)
            changed[0]['stamp'][0] += 1
            with self.assertRaises(ValueError):
                cache.prepare(root/'inputs.sqlite', changed, 'test', 1, NoStop())


if __name__ == '__main__':
    unittest.main()
