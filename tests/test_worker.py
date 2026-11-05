"""Regression checks for engine configuration; these do not run CUDA inference."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class WorkerTests(unittest.TestCase):
    def load_worker(self, name='inference'):
        source = Path(__file__).resolve().parents[1]/'workers'/f'{name}.py'
        spec = importlib.util.spec_from_file_location('voiceai_'+name+'_test', source)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ):
            spec.loader.exec_module(worker)
        return worker

    def test_tts_avoids_fp16_on_turing_without_changing_vc(self):
        # No torch/Qwen installation or network is needed for this contract test.
        source = Path(__file__).resolve().parents[1]/'workers'/'inference.py'
        spec = importlib.util.spec_from_file_location('voiceai_worker_test', source)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ):
            spec.loader.exec_module(worker)
        fp32 = object()
        loader = Mock()
        with patch.dict(sys.modules, {
            'torch': SimpleNamespace(float32=fp32),
            'qwen_tts': SimpleNamespace(Qwen3TTSModel=loader),
        }):
            model = worker.qwen_model(Path('existing-local-model'))
        self.assertIs(model, loader.from_pretrained.return_value)
        loader.from_pretrained.assert_called_once_with(
            'existing-local-model', device_map='cuda:0', dtype=fp32,
            attn_implementation='sdpa')
        self.assertTrue(worker.seed_args().fp16)

    def test_legacy_partial_transcript_is_never_sent_to_qwen(self):
        worker = self.load_worker()
        model = Mock()
        model.generate_voice_clone.return_value = ([[.1] * 20], 10)
        request = {'text':'今日天気いいね', 'reference':'reference.wav', 'transcript':'途中のみ'}
        with patch.dict(sys.modules, {'numpy': fake_numpy()}):
            _, _, mode, budget = worker.generate_tts(model, request)
        self.assertEqual(mode, 'speaker_embedding')
        self.assertLess(budget / 12.5, 30)
        self.assertIsNone(model.generate_voice_clone.call_args.kwargs['ref_text'])
        self.assertTrue(model.generate_voice_clone.call_args.kwargs['x_vector_only_mode'])

    def test_runaway_retries_text_free_once_then_fails(self):
        worker = self.load_worker()
        model = Mock()
        model.generate_voice_clone.return_value = ([[.1] * 3276], 10)
        request = {'text':'今日天気いいね', 'reference':'reference.wav', 'transcript':'自動認識',
                   'transcript_source':'automatic'}
        with patch.dict(sys.modules, {'numpy': fake_numpy()}):
            with self.assertRaisesRegex(RuntimeError, '正常に終了しなかった'):
                worker.generate_tts(model, request)
        self.assertEqual(model.generate_voice_clone.call_count, 2)
        first, second = model.generate_voice_clone.call_args_list
        self.assertEqual(first.kwargs['ref_text'], '自動認識')
        self.assertFalse(first.kwargs['x_vector_only_mode'])
        self.assertIsNone(second.kwargs['ref_text'])
        self.assertTrue(second.kwargs['x_vector_only_mode'])

    def test_reference_cache_reuses_and_invalidates_by_audio_and_revision(self):
        worker = self.load_worker('reference')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'models').mkdir()
            model_path = root/'models'/'snapshot'
            model_path.mkdir()
            manifest = root/'models'/'asr-ready.json'
            manifest.write_text(json.dumps({'path':str(model_path), 'revision':'rev1'}))
            reference = root/'reference.wav'
            reference.write_bytes(b'first audio')
            request = {'reference':str(reference), 'output':str(root/'output.wav')}
            gpu = Mock()
            with patch.object(worker, 'recognize', return_value=('自動認識した全文', 'recognized')) as recognize:
                worker.transcribe(root, request, gpu)
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 1)
                self.assertTrue(json.loads((root/'reference-analysis.json').read_text())['cache_hit'])
                reference.write_bytes(b'changed audio')
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 2)
                manifest.write_text(json.dumps({'path':str(model_path), 'revision':'rev2'}))
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 3)
                (root/'auto-transcript.json').write_text('broken cache')
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 4)
            self.assertEqual(gpu.call_count, 4)

    def test_empty_recognition_uses_no_text_and_is_not_cached(self):
        worker = self.load_worker('reference')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'models').mkdir()
            (root/'models'/'asr-ready.json').write_text(json.dumps({'path':str(root), 'revision':'rev1'}))
            (root/'reference.wav').write_bytes(b'audio')
            request = {'reference':str(root/'reference.wav'), 'output':str(root/'output.wav')}
            with patch.object(worker, 'recognize', return_value=('', 'empty_or_truncated')):
                worker.transcribe(root, request, Mock())
            self.assertFalse((root/'auto-transcript.json').exists())
            self.assertEqual(json.loads((root/'reference-analysis.json').read_text())['text'], '')


def fake_numpy():
    # Only array plumbing is replaced; tests exercise our retry/limit logic.
    class Samples(list):
        @property
        def size(self):
            return len(self)
    return SimpleNamespace(asarray=Samples, isfinite=lambda a: SimpleNamespace(all=lambda: all(math.isfinite(v) for v in a)))


if __name__ == '__main__':
    unittest.main()
