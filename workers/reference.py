"""Local reference transcription. Runs in the VC environment, before TTS."""
import hashlib
import json
import array
import math
import sys
import wave
from pathlib import Path

ASR_MODEL_ID = 'openai/whisper-small'
CACHE_VERSION = 2


def prepare_reference(reference):
    """Shorten only long, near-silent gaps; keep the uploaded reference intact.

    ASR and Qwen must consume the exact same derived WAV. This is an amplitude
    gate, not speaker separation or speech recognition; do not remove short gaps.
    """
    with wave.open(str(reference), 'rb') as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError('VOICEAI:参照音声の形式を確認できません。ログを共有してください。')
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = array.array('h', raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    step = max(1, round(rate * .02))
    threshold = 32768 * 10 ** (-75 / 20)
    quiet = []
    for start in range(0, len(samples), step):
        frame = samples[start:start+step]
        quiet.append(math.sqrt(sum(v*v for v in frame) / len(frame)) < threshold)
    removals = []
    start = None
    for i, silent in enumerate(quiet + [False]):
        if silent and start is None:
            start = i * step
        elif not silent and start is not None:
            end = min(i * step, len(samples))
            if end - start >= .8 * rate:
                left = start + (0 if start == 0 else round(.12 * rate))
                right = end - (0 if end == len(samples) else round(.12 * rate))
                # Retain a short natural tail even for a long final silence.
                if end == len(samples) and start > 0:
                    left = start + round(.2 * rate)
                removals.append((left, right))
            start = None
    chunks, cursor = [], 0
    for left, right in removals:
        chunks.append(raw[cursor*2:left*2])
        cursor = right
    chunks.append(raw[cursor*2:])
    cleaned = b''.join(chunks)
    # Very sparse samples need more care than this conservative automatic gate.
    if len(cleaned) < 3 * rate * 2:
        cleaned = raw
    target = reference.with_name('tts-reference.wav')
    temporary = target.with_suffix('.tmp')
    with wave.open(str(temporary), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(cleaned)
    temporary.replace(target)
    details = {'original_seconds': len(raw) / (2 * rate),
               'prepared_seconds': len(cleaned) / (2 * rate),
               'removed_silence_seconds': (len(raw) - len(cleaned)) / (2 * rate)}
    return target, details


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_model(path):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(path), torch_dtype=torch.float32, attn_implementation='sdpa',
        local_files_only=True).to('cuda:0').eval()
    return processor, model


def prepare(root):
    from huggingface_hub import snapshot_download
    # Reuses Whisper weights cached by Seed-VC; also obtains text tokenizer files.
    path = snapshot_download(ASR_MODEL_ID, allow_patterns=['*.json', '*.txt', 'model.safetensors'])
    processor, model = load_model(path)
    return {'model': ASR_MODEL_ID, 'path': path, 'revision': Path(path).name,
            'precision': 'float32'}


def recognize(reference, model_path):
    import librosa
    import torch
    processor, model = load_model(model_path)
    audio, rate = librosa.load(str(reference), sr=16000, mono=True)
    if not 0 < len(audio) <= 30 * rate:
        raise RuntimeError('VOICEAI:参照音声の長さを確認できません。ログを共有してください。')
    inputs = processor(audio, sampling_rate=rate, return_tensors='pt', return_attention_mask=True)
    with torch.inference_mode():
        tokens = model.generate(
            input_features=inputs.input_features.to('cuda:0', dtype=torch.float32),
            attention_mask=inputs.attention_mask.to('cuda:0'),
            language='ja', task='transcribe', return_timestamps=False,
            do_sample=False, num_beams=1, max_new_tokens=440)
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    # Avoid feeding an incomplete transcript to the clone model.
    if tokens.shape[-1] >= 440 or not text or len(text) > 1000:
        return '', 'empty_or_truncated'
    return text, 'recognized'


def transcribe(root, request, check_gpu):
    manifest_path = root/'models'/'asr-ready.json'
    if not manifest_path.is_file():
        raise RuntimeError('VOICEAI:自動音声認識の初回準備が必要です。アプリを終了し、START.batを実行してください。')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    reference = Path(request['reference'])
    original_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    prepared, preparation = prepare_reference(reference)
    fingerprint = hashlib.sha256(prepared.read_bytes()).hexdigest()
    identity = {'schema': CACHE_VERSION, 'audio_sha256': fingerprint, 'original_sha256': original_hash,
                'model': ASR_MODEL_ID, 'revision': manifest['revision'], 'language': 'ja'}
    cache_path = reference.with_name('auto-transcript.json')
    try:
        cached = json.loads(cache_path.read_text('utf-8'))
    except (OSError, ValueError):
        cached = None
    hit = (isinstance(cached, dict) and all(cached.get(k) == v for k, v in identity.items())
           and isinstance(cached.get('text'), str) and 0 < len(cached['text']) <= 1000
           and cached.get('status') == 'recognized')
    if hit:
        result = {**cached, 'cache_hit': True}
    else:
        check_gpu()
        if not Path(manifest['path']).is_dir():
            raise RuntimeError('VOICEAI:音声認識モデルが見つかりません。02_prepare_models.batを実行してください。')
        text, status = recognize(prepared, manifest['path'])
        result = {**identity, 'text': text, 'status': status, 'cache_hit': False}
        if text:
            save_json(cache_path, result)
    result.update({'reference': str(prepared), 'preparation': preparation})
    save_json(Path(request['output']).parent/'reference-analysis.json', result)
    print('Reference preparation: ' + json.dumps(preparation), flush=True)
    print('Reference analysis: ' + result['status'] + ', cache_hit=' + str(result['cache_hit']), flush=True)
