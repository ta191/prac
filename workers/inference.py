"""Real inference worker. Each invocation exits and releases CPU/GPU memory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ.get('VOICEAI_ROOT', Path(__file__).resolve().parent.parent)).resolve()
os.environ.setdefault('HF_HOME', str(ROOT/'models'/'huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'models'/'torch'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['PATH'] = str(ROOT/'tools') + os.pathsep + os.environ.get('PATH', '')
MODEL_ID = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
SEED_REVISION = '51383efd921027683c89e5348211d93ff12ac2a8'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('VOICEAI:GPUを使用できません。04_diagnose.batの結果を確認してください。')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    return {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'vram_bytes': torch.cuda.get_device_properties(0).total_memory}


def qwen_model(path):
    import torch
    from qwen_tts import Qwen3TTSModel
    # Qwen's code predictor can overflow in FP16. Turing lacks native BF16.
    # Upstream recommends FP32: https://github.com/QwenLM/Qwen3-TTS/issues/43
    print('TTS runtime: device=cuda:0, precision=float32, attention=sdpa', flush=True)
    return Qwen3TTSModel.from_pretrained(str(path), device_map='cuda:0',
                                       dtype=torch.float32, attn_implementation='sdpa')


def seed_module():
    vendor = ROOT/'vendor'/'seed-vc'
    if not (vendor/'inference.py').is_file():
        raise RuntimeError('VOICEAI:Seed-VCがありません。01_setup.batを実行してください。')
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)
    import inference
    return inference


def seed_args(request=None):
    request = request or {}
    output = Path(request.get('output', ROOT/'data'/'unused.wav'))
    return argparse.Namespace(source=request.get('source'), target=request.get('reference'),
                              output=str(output.parent/'seed-output'), diffusion_steps=request.get('steps', 25),
                              length_adjust=1.0, inference_cfg_rate=.7, f0_condition=False,
                              auto_f0_adjust=False, semi_tone_shift=0,
                              checkpoint=None, config=None, fp16=True)


def prepare(kind):
    gpu = check_gpu()
    manifest = ROOT/'models'/f'{kind}-ready.json'
    manifest.unlink(missing_ok=True)
    if kind == 'tts':
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL_ID)
        model = qwen_model(path)  # Also downloads tokenizer dependencies, and checks CUDA load.
        details = {'model': MODEL_ID, 'path': path, 'revision': Path(path).name,
                   'precision': 'float32'}
        del model
    else:
        module = seed_module()
        models = module.load_models(seed_args())
        # Record resolved checkpoint snapshots used by the pinned upstream loader.
        snapshots = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'vendor'/'seed-vc'/'checkpoints').rglob('snapshots/*') if p.is_dir())
        details = {'model': 'seed-uvit-whisper-small-wavenet', 'code_revision': SEED_REVISION,
                   'snapshots': snapshots}
        del models
    import torch
    details.update(gpu)
    details['peak_vram_mib'] = round(torch.cuda.max_memory_allocated()/1024**2)
    details['prepared_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write_json(manifest, details)
    print(f'{kind}: download and GPU loading completed. Speech quality is not yet tested.', flush=True)


def infer(kind, request):
    gpu = check_gpu()
    print('GPU runtime: ' + json.dumps(gpu, ensure_ascii=True), flush=True)
    output = Path(request['output'])
    started = time.monotonic()
    if kind == 'tts':
        import numpy as np
        import soundfile as sf
        manifest = json.loads((ROOT/'models'/'tts-ready.json').read_text('utf-8'))
        if not Path(manifest['path']).is_dir():
            raise RuntimeError('VOICEAI:モデルの保存先が見つかりません。02_prepare_models.batを再実行してください。')
        model = qwen_model(manifest['path'])
        reference_text = request['transcript'].strip()
        print('TTS stage: generating voice clone', flush=True)
        audio, rate = model.generate_voice_clone(
            text=request['text'], language='Japanese', ref_audio=request['reference'],
            ref_text=reference_text or None, x_vector_only_mode=not bool(reference_text),
            max_new_tokens=4096)
        samples = np.asarray(audio[0])
        if not np.isfinite(samples).all() or samples.size == 0:
            raise RuntimeError('VOICEAI:音声生成が不安定でした。文章を短くして再実行してください。')
        sf.write(output, samples, rate, subtype='PCM_16')
    else:
        import soundfile as sf
        module = seed_module()
        args = seed_args(request)
        module.main(args)
        matches = list(Path(args.output).glob('*.wav'))
        if len(matches) != 1:
            raise RuntimeError('VOICEAI:音声変換の出力を確認できませんでした。ログを確認してください。')
        samples, rate = sf.read(matches[0])
        sf.write(output, samples, rate, subtype='PCM_16')
        matches[0].unlink()
        Path(args.output).rmdir()
    import torch
    gpu.update({'inference_seconds': round(time.monotonic()-started, 2),
                'audio_seconds': round(len(samples)/rate, 2),
                'peak_vram_mib': round(torch.cuda.max_memory_allocated()/1024**2),
                'precision': 'float32' if kind == 'tts' else 'float16', 'kind': kind})
    write_json(output.parent/'metrics.json', gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['tts', 'vc'])
    parser.add_argument('request', nargs='?')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare(args.kind)
        else:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            if not args.request:
                parser.error('request JSON is required')
            infer(args.kind, json.loads(Path(args.request).read_text('utf-8')))
    except Exception as exc:
        traceback.print_exc()
        if 'device-side assert' in str(exc).lower():
            print('VOICEAI:GPUの音声生成処理が異常終了しました。画面の「ログ」を保存して共有してください。モデルの再取得ではなく、実行設定の確認が必要です。', flush=True)
        sys.exit(1)
