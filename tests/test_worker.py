"""Regression checks for engine configuration; these do not run CUDA inference."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import wave
import struct
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
            with self.assertRaisesRegex(RuntimeError, '自動認識を確認できません'):
                worker.generate_tts(model, request)
        model.generate_voice_clone.assert_not_called()

    def test_runaway_retries_same_voice_mode_once_then_fails(self):
        worker = self.load_worker()
        model = Mock()
        model.generate_voice_clone.return_value = ([[.1] * 3276], 10)
        request = {'text':'今日天気いいね', 'reference':'reference.wav', 'transcript':'自動認識',
                   'transcript_source':'automatic'}
        with patch.dict(sys.modules, {'numpy': fake_numpy()}), patch.object(worker, 'seed_generation') as seed:
            with self.assertRaisesRegex(RuntimeError, '正常に終了しなかった'):
                worker.generate_tts(model, request)
        self.assertEqual([c.args[0] for c in seed.call_args_list], [42, 43])
        self.assertEqual(model.generate_voice_clone.call_count, 2)
        first, second = model.generate_voice_clone.call_args_list
        self.assertEqual(first.kwargs['ref_text'], '自動認識')
        self.assertFalse(first.kwargs['x_vector_only_mode'])
        self.assertEqual(second.kwargs['ref_text'], '自動認識')
        self.assertFalse(second.kwargs['x_vector_only_mode'])

    def test_same_request_resets_seed_and_uses_full_text(self):
        worker = self.load_worker()
        model = Mock()
        model.generate_voice_clone.return_value = ([[.1] * 20], 10)
        request = {'text':'今日天気いいね', 'reference':'reference.wav', 'transcript':'自動認識',
                   'transcript_source':'automatic'}
        with patch.dict(sys.modules, {'numpy': fake_numpy()}), patch.object(worker, 'seed_generation') as seed:
            first = worker.generate_tts(model, request)
            second = worker.generate_tts(model, request)
        self.assertEqual(first, second)
        self.assertEqual([c.args[0] for c in seed.call_args_list], [42, 42])
        self.assertTrue(model.generate_voice_clone.call_args.kwargs['non_streaming_mode'])
        self.assertTrue(model.generate_voice_clone.call_args.kwargs['do_sample'])
        self.assertLess(first[3]/12.5, 30)

    def test_long_silences_are_shortened_without_changing_speech_or_original(self):
        worker = self.load_worker('reference')
        rate = 24000
        tone = b''.join(struct.pack('<h', int(2000*math.sin(2*math.pi*220*i/rate))) for i in range(2*rate))
        # A quiet, 0.4s consonant-like region must also survive unchanged.
        quiet = struct.pack('<h', 10) * int(.4*rate)
        raw = b'\0\0' * rate + tone + quiet + b'\0\0' * (3*rate) + tone + b'\0\0' * rate
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder)/'reference.wav'
            write_reference(reference, raw=raw)
            original = reference.read_bytes()
            prepared, details = worker.prepare_reference(reference)
            self.assertEqual(reference.read_bytes(), original)
            self.assertGreater(details['removed_silence_seconds'], 4)
            with wave.open(str(prepared), 'rb') as wav:
                cleaned = wav.readframes(wav.getnframes())
            self.assertEqual(cleaned.count(tone), 2)
            self.assertIn(quiet, cleaned)
            # Short pauses and quiet audio must remain even between loud regions.
            write_reference(reference, raw=tone + quiet + tone)
            prepared, details = worker.prepare_reference(reference)
            with wave.open(str(prepared), 'rb') as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), tone + quiet + tone)
            self.assertEqual(details['removed_silence_seconds'], 0)

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
            write_reference(reference)
            request = {'reference':str(reference), 'output':str(root/'output.wav')}
            gpu = Mock()
            with patch.object(worker, 'recognize', return_value=('自動認識した全文', 'recognized')) as recognize:
                worker.transcribe(root, request, gpu)
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 1)
                self.assertEqual(recognize.call_args.args[0], reference.with_name('tts-reference.wav'))
                self.assertTrue(json.loads((root/'reference-analysis.json').read_text())['cache_hit'])
                write_reference(reference, amplitude=1500)
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 2)
                manifest.write_text(json.dumps({'path':str(model_path), 'revision':'rev2'}))
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 3)
                (root/'auto-transcript.json').write_text('broken cache')
                worker.transcribe(root, request, gpu)
                self.assertEqual(recognize.call_count, 4)
            self.assertEqual(gpu.call_count, 4)

    def test_empty_recognition_is_not_cached(self):
        worker = self.load_worker('reference')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'models').mkdir()
            (root/'models'/'asr-ready.json').write_text(json.dumps({'path':str(root), 'revision':'rev1'}))
            write_reference(root/'reference.wav')
            request = {'reference':str(root/'reference.wav'), 'output':str(root/'output.wav')}
            with patch.object(worker, 'recognize', return_value=('', 'empty_or_truncated')):
                worker.transcribe(root, request, Mock())
            self.assertFalse((root/'auto-transcript.json').exists())
            self.assertEqual(json.loads((root/'reference-analysis.json').read_text())['text'], '')


def write_reference(path, amplitude=2000, raw=None):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        if raw is None:
            raw = b''.join(struct.pack('<h',int(amplitude*math.sin(2*math.pi*220*i/24000))) for i in range(76800))
        wav.writeframes(raw)


def fake_numpy():
    # Only array plumbing is replaced; tests exercise our retry/limit logic.
    class Samples(list):
        @property
        def size(self):
            return len(self)
    return SimpleNamespace(asarray=Samples, isfinite=lambda a: SimpleNamespace(all=lambda: all(math.isfinite(v) for v in a)))


if __name__ == '__main__':
    unittest.main()
