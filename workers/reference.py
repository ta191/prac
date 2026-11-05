"""Local reference transcription. Runs in the VC environment, before TTS."""
import hashlib
import json
from pathlib import Path

ASR_MODEL_ID = 'openai/whisper-small'
CACHE_VERSION = 1


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
    # Avoid feeding an incomplete transcript to the clone model. A text-free
    # speaker embedding is supported by Qwen and needs no user intervention.
    if tokens.shape[-1] >= 440 or not text or len(text) > 1000:
        return '', 'empty_or_truncated'
    return text, 'recognized'


def transcribe(root, request, check_gpu):
    manifest_path = root/'models'/'asr-ready.json'
    if not manifest_path.is_file():
        raise RuntimeError('VOICEAI:自動音声認識の初回準備が必要です。アプリを終了し、START.batを実行してください。')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    reference = Path(request['reference'])
    fingerprint = hashlib.sha256(reference.read_bytes()).hexdigest()
    identity = {'schema': CACHE_VERSION, 'audio_sha256': fingerprint,
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
        text, status = recognize(reference, manifest['path'])
        result = {**identity, 'text': text, 'status': status, 'cache_hit': False}
        if text:
            save_json(cache_path, result)
    save_json(Path(request['output']).parent/'reference-analysis.json', result)
    print('Reference analysis: ' + result['status'] + ', cache_hit=' + str(result['cache_hit']), flush=True)
