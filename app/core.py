from __future__ import annotations

import array
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
import wave
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class UserError(Exception):
    pass


def identity(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise UserError('項目のIDが正しくありません。')


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data = self.root / 'data'
        for name in ('voices', 'jobs', 'tmp'):
            (self.data / name).mkdir(parents=True, exist_ok=True)
        self.db = self.data / 'voiceai.sqlite3'
        self.lock = threading.RLock()
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS voices (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, transcript TEXT NOT NULL,
                    duration REAL NOT NULL, created TEXT NOT NULL, consent INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, voice_id TEXT NOT NULL,
                    voice_name TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
                    message TEXT NOT NULL, created TEXT NOT NULL, finished TEXT,
                    seconds REAL, metrics TEXT);
            ''')
            con.execute("UPDATE jobs SET status='failed', message=?, finished=? WHERE status IN ('queued','running')",
                        ('前回の終了で処理が中断されました。再実行してください。', now()))

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def rows(self, table):
        assert table in ('voices', 'jobs')
        with self.connect() as con:
            return [dict(row) for row in con.execute(f'SELECT * FROM {table} ORDER BY created DESC')]

    def get(self, table, key):
        assert table in ('voices', 'jobs')
        key = identity(key)
        with self.connect() as con:
            row = con.execute(f'SELECT * FROM {table} WHERE id=?', (key,)).fetchone()
        if not row:
            raise UserError('項目が見つかりません。画面を更新してください。')
        return dict(row)

    def voice_path(self, key):
        return self.data / 'voices' / identity(key) / 'reference.wav'

    def job_path(self, key):
        return self.data / 'jobs' / identity(key)

    def update_job(self, key, status, message, **fields):
        assert set(fields) <= {'finished', 'seconds', 'metrics'}
        args = {'status': status, 'message': message, **fields}
        with self.connect() as con:
            con.execute('UPDATE jobs SET '+', '.join(f'{k}=?' for k in args)+' WHERE id=?',
                        (*args.values(), identity(key)))


def ffmpeg(root):
    bundled = root / 'tools' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    found = str(bundled) if bundled.is_file() else shutil.which('ffmpeg')
    if not found:
        raise UserError('FFmpegがありません。01_setup.batを実行してください。')
    return found


def audio_info(path):
    with wave.open(str(path), 'rb') as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise UserError('音声形式を確認できませんでした。')
        duration = audio.getnframes() / audio.getframerate()
        samples = array.array('h', audio.readframes(audio.getnframes()))
    import sys
    if sys.byteorder != 'little':
        samples.byteswap()
    values = samples[::8]
    rms = math.sqrt(sum(v*v for v in values) / max(1, len(values))) / 32768
    return duration, rms


def normalize(root, source, output, extension, minimum, maximum):
    if extension not in ('.wav', '.mp3'):
        raise UserError('WAVまたはMP3ファイルを選んでください。')
    command = [ffmpeg(root), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-protocol_whitelist', 'file,pipe', '-f', extension[1:], '-i', str(source),
               '-t', str(maximum + 1), '-vn', '-ac', '1', '-ar', '24000',
               '-map_metadata', '-1', '-c:a', 'pcm_s16le', str(output)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=90)
        duration, rms = audio_info(output)
    except (subprocess.SubprocessError, wave.Error, EOFError):
        raise UserError('音声を読み込めません。ファイルの破損や形式を確認してください。')
    if not minimum <= duration <= maximum:
        raise UserError(f'音声の長さは{minimum}〜{maximum}秒にしてください（現在 {duration:.1f}秒）。')
    if rms < 0.0002:
        raise UserError('音声が無音、または小さすぎます。明瞭な録音を選んでください。')
    return duration


def runtime_paths(root):
    folder = 'Scripts' if os.name == 'nt' else 'bin'
    executable = 'python.exe' if os.name == 'nt' else 'python'
    return {kind: root / f'.venv-{kind}' / folder / executable for kind in ('tts', 'vc')}


def runtime_status(root):
    paths = runtime_paths(root)
    try:
        ffmpeg(root)
        audio_ready = True
    except UserError:
        audio_ready = False
    prepared = {}
    for kind in ('tts', 'vc'):
        manifest = root / 'models' / f'{kind}-ready.json'
        prepared[kind] = paths[kind].is_file() and manifest.is_file()
    asr_ready = paths['vc'].is_file() and (root/'models'/'asr-ready.json').is_file()
    return {'tts': prepared['tts'] and asr_ready, 'vc': prepared['vc'], 'asr': asr_ready, 'audio': audio_ready,
            'message': '利用準備完了' if all(prepared.values()) and asr_ready and audio_ready else
                       '初回準備・更新が必要です。アプリを終了し、START.batを実行してください。',
            'version': '0.2.1', 'offline_inference': True}


def worker_environment(root):
    env = os.environ.copy()
    env.update({'VOICEAI_ROOT': str(root), 'HF_HOME': str(root / 'models' / 'huggingface'),
                'TORCH_HOME': str(root / 'models' / 'torch'), 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                'GRADIO_ANALYTICS_ENABLED': 'False', 'PYTHONUTF8': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'})
    return env


class JobManager:
    def __init__(self, store, runner=None):
        self.store = store
        self.runner = runner or self.run_worker
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.cancelled = set()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=12)

    def submit(self, kind, voice_id, text='', source=None, steps=25):
        with self.store.lock:
            voice = self.store.get('voices', voice_id)
            active = [j for j in self.store.rows('jobs') if j['status'] in ('queued', 'running')]
            if len(active) >= 8:
                raise UserError('待機中の処理が多いため、完了してから再度お試しください。')
            key = str(uuid.uuid4())
            folder = self.store.job_path(key)
            folder.mkdir()
            request = {'id': key, 'kind': kind, 'reference': str(self.store.voice_path(voice_id)),
                       'transcript': '', 'transcript_source': 'automatic_pending', 'text': text,
                       'output': str(folder / 'output.wav'), 'steps': steps,
                       'source': str(folder / 'source.wav') if source else None}
            try:
                if source:
                    shutil.copyfile(source, folder / 'source.wav')
                (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
                with self.store.connect() as con:
                    con.execute('INSERT INTO jobs (id,kind,voice_id,voice_name,text,status,message,created) VALUES (?,?,?,?,?,?,?,?)',
                                (key, kind, voice_id, voice['name'], text, 'queued', '順番待ち', now()))
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            self.wake.set()
            return key

    def cancel(self, key):
        with self.store.lock:
            job = self.store.get('jobs', key)
            if job['status'] in ('queued', 'running'):
                if job['status'] == 'running':
                    self.cancelled.add(key)
                self.store.update_job(key, 'cancelled', 'キャンセルしました。', finished=now())
                self.wake.set()

    def loop(self):
        while not self.stop.is_set():
            with self.store.lock:
                waiting = [j for j in reversed(self.store.rows('jobs')) if j['status'] == 'queued']
                if waiting:
                    job = waiting[0]
                    self.store.update_job(job['id'], 'running', 'AI処理中です。初回の読み込みには時間がかかります。')
                else:
                    job = None
            if not job:
                self.wake.wait(.5)
                self.wake.clear()
                continue
            started = time.monotonic()
            try:
                self.runner(job, self)
                with self.store.lock:
                    if job['id'] not in self.cancelled and not self.stop.is_set():
                        folder = self.store.job_path(job['id'])
                        if not (folder/'output.wav').is_file() or not (folder/'output.mp3').is_file():
                            raise UserError('出力ファイルが作成されませんでした。ログを確認してください。')
                        metrics = (folder/'metrics.json').read_text('utf-8') if (folder/'metrics.json').is_file() else '{}'
                        self.store.update_job(job['id'], 'done', '音声が完成しました。', finished=now(),
                                              seconds=round(time.monotonic()-started, 2), metrics=metrics)
                    elif self.stop.is_set() and job['id'] not in self.cancelled:
                        self.store.update_job(job['id'], 'failed', 'アプリの終了で処理が中断されました。', finished=now())
            except Exception as error:
                with self.store.lock:
                    if job['id'] not in self.cancelled:
                        message = str(error) if isinstance(error, UserError) else '処理中にエラーが発生しました。ログを確認してください。'
                        self.store.update_job(job['id'], 'failed', message, finished=now())
            finally:
                with self.store.lock:
                    if self.store.get('jobs', job['id'])['status'] != 'done':
                        for filename in ('output.wav', 'output.mp3'):
                            (self.store.job_path(job['id']) / filename).unlink(missing_ok=True)
                    self.cancelled.discard(job['id'])

    def interrupted(self, key):
        with self.store.lock:
            return self.stop.is_set() or key in self.cancelled

    def run_worker(self, job, manager):
        root = self.store.root
        folder = self.store.job_path(job['id'])
        python = runtime_paths(root)[job['kind']]
        if not python.is_file():
            raise UserError('AI実行環境がありません。01_setup.batを実行してください。')
        if job['kind'] == 'tts':
            asr_python = runtime_paths(root)['vc']
            if not asr_python.is_file():
                raise UserError('自動音声認識の準備が必要です。アプリを終了し、START.batを実行してください。')
            with self.store.lock:
                if self.interrupted(job['id']):
                    return
                self.store.update_job(job['id'], 'running', '登録した声を自動解析しています。初回のみ少し時間がかかります。')
            # A separate process finishes and releases the GPU before Qwen loads.
            self.run_process([str(asr_python), str(root/'workers'/'inference.py'), 'asr', str(folder/'request.json')],
                             job['id'], folder/'worker.log', worker_environment(root), timeout=300)
            if self.interrupted(job['id']):
                return
            analysis = json.loads((folder/'reference-analysis.json').read_text('utf-8'))
            request_path = folder/'request.json'
            request = json.loads(request_path.read_text('utf-8'))
            request['reference'] = analysis['reference']
            request['transcript'] = analysis['text']
            request['transcript_source'] = 'automatic'
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            with self.store.lock:
                if self.interrupted(job['id']):
                    return
                self.store.update_job(job['id'], 'running', '声の解析が完了しました。音声を生成しています。')
        command = [str(python), str(root/'workers'/'inference.py'), job['kind'], str(folder/'request.json')]
        self.run_process(command, job['id'], folder/'worker.log', worker_environment(root), timeout=1800)
        if self.interrupted(job['id']):
            return
        duration, rms = audio_info(folder/'output.wav')
        if duration <= 0 or rms < .00002:
            raise UserError('生成結果が無音でした。参照音声や文章を変えてお試しください。')
        self.run_process([ffmpeg(root), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                          '-i', str(folder/'output.wav'), '-map_metadata', '-1', '-codec:a', 'libmp3lame',
                          '-b:a', '192k', str(folder/'output.mp3')], job['id'], folder/'worker.log',
                         worker_environment(root), timeout=90)

    def run_process(self, command, key, logpath, env, timeout):
        started = time.monotonic()
        with logpath.open('ab') as log:
            process = subprocess.Popen(command, cwd=self.store.root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                while process.poll() is None:
                    if self.interrupted(key):
                        return
                    if time.monotonic() - started > timeout:
                        raise UserError('処理時間の上限を超えました。入力を短くして再実行してください。')
                    time.sleep(.2)
                if process.returncode:
                    content = logpath.read_text('utf-8', errors='replace')[-12000:]
                    if 'out of memory' in content.lower():
                        raise UserError('GPUメモリが不足しました。他のGPU使用アプリを閉じ、入力を短くしてください。')
                    if 'VOICEAI:' in content:
                        raise UserError(content.rsplit('VOICEAI:', 1)[-1].splitlines()[0][:300])
                    raise UserError('AI処理に失敗しました。「ログ」を確認してください。モデル準備が未完了なら02_prepare_models.batを実行してください。')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
