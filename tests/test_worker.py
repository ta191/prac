"""Regression checks for engine configuration; these do not run CUDA inference."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class WorkerTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
