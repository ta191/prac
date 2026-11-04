from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import shutil
import uuid

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .core import Store, JobManager, UserError, normalize, runtime_status, now

SOURCE = Path(__file__).resolve().parent.parent


class Rename(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    transcript: str = Field(default='', max_length=1000)


class TTSRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=1, max_length=500)


async def receive_audio(store, file, minimum, maximum):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ('.wav', '.mp3'):
        raise UserError('WAVまたはMP3ファイルを選んでください。')
    folder = store.data / 'tmp' / str(uuid.uuid4())
    folder.mkdir()
    try:
        total = 0
        with (folder/'input.bin').open('wb') as out:
            while chunk := await file.read(1024*1024):
                total += len(chunk)
                if total > 40*1024*1024:
                    raise UserError('ファイルは40 MB以内にしてください。')
                out.write(chunk)
        duration = await run_in_threadpool(normalize, store.root, folder/'input.bin', folder/'audio.wav',
                                         suffix, minimum, maximum)
        return folder, duration
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        await file.close()


def create_app(root=None, runner=None):
    root = Path(root or os.environ.get('VOICEAI_ROOT', SOURCE))
    store = Store(root)
    manager = JobManager(store, runner)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.manager, app.state.token = store, manager, token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.middleware('http')
    async def protect(request: Request, call_next):
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'この操作はアプリの画面から実行してください。'}, status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            if not secrets.compare_digest(request.headers.get('x-voiceai-token', ''), token):
                return JSONResponse({'detail': '画面を更新してから再度お試しください。'}, status_code=403)
            try:
                if int(request.headers.get('content-length', '0')) > 42*1024*1024:
                    return JSONResponse({'detail': 'ファイルは40 MB以内にしてください。'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail': 'リクエストが正しくありません。'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.exception_handler(UserError)
    async def user_error(request, error):
        return JSONResponse({'detail': str(error)}, status_code=400)

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (SOURCE/'web'/'index.html').read_text('utf-8').replace('__TOKEN__', token)

    @app.get('/assets/{name}')
    def asset(name: str):
        if name not in ('app.js', 'style.css'):
            return JSONResponse({'detail': '見つかりません。'}, status_code=404)
        return FileResponse(SOURCE/'web'/name)

    @app.get('/api/status')
    def status():
        return runtime_status(store.root)

    @app.get('/api/voices')
    def voices():
        return store.rows('voices')

    @app.post('/api/voices')
    async def create_voice(name: str = Form(..., min_length=1, max_length=40),
                           transcript: str = Form('', max_length=1000),
                           consent: bool = Form(...), file: UploadFile = File(...)):
        if not name.strip():
            raise UserError('ボイス名を入力してください。')
        if not consent:
            raise UserError('この音声を利用する許可を確認してください。')
        temporary, duration = await receive_audio(store, file, 3, 30)
        key = str(uuid.uuid4())
        target = store.voice_path(key)
        try:
            with store.lock:
                target.parent.mkdir()
                shutil.copyfile(temporary/'audio.wav', target)
                with store.connect() as con:
                    con.execute('INSERT INTO voices VALUES (?,?,?,?,?,?)',
                                (key, name.strip(), transcript.strip(), duration, now(), 1))
        except Exception:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return store.get('voices', key)

    @app.put('/api/voices/{key}')
    def rename_voice(key: str, value: Rename):
        if not value.name.strip():
            raise UserError('ボイス名を入力してください。')
        with store.lock:
            voice = store.get('voices', key)
            with store.connect() as con:
                con.execute('UPDATE voices SET name=?,transcript=? WHERE id=?',
                            (value.name.strip(), value.transcript.strip(), voice['id']))
        return store.get('voices', key)

    @app.delete('/api/voices/{key}')
    def delete_voice(key: str):
        with store.lock:
            voice = store.get('voices', key)
            if any(j['voice_id'] == key and j['status'] in ('queued', 'running') for j in store.rows('jobs')):
                raise UserError('このボイスは処理中です。完了またはキャンセル後に削除してください。')
            # Cancellation can still be releasing the file; protect the active worker too.
            if key in {store.get('jobs', j)['voice_id'] for j in manager.cancelled}:
                raise UserError('キャンセル処理の終了を待ってから削除してください。')
            shutil.rmtree(store.voice_path(key).parent)
            with store.connect() as con:
                con.execute('DELETE FROM voices WHERE id=?', (voice['id'],))
        return {'ok': True}

    @app.get('/api/voices/{key}/audio')
    def reference(key: str):
        store.get('voices', key)
        return FileResponse(store.voice_path(key), media_type='audio/wav')

    @app.get('/api/jobs')
    def jobs():
        return store.rows('jobs')[:100]

    @app.post('/api/jobs/tts')
    def create_tts(value: TTSRequest):
        if not value.text.strip():
            raise UserError('読み上げる文章を入力してください。')
        if not runtime_status(store.root)['tts']:
            raise UserError('読み上げモデルの準備が必要です。02_prepare_models.batを実行してください。')
        return {'id': manager.submit('tts', value.voice_id, text=value.text.strip())}

    @app.post('/api/jobs/vc')
    async def create_vc(voice_id: str = Form(...), steps: int = Form(25, ge=10, le=50),
                        file: UploadFile = File(...)):
        store.get('voices', voice_id)
        if not runtime_status(store.root)['vc']:
            raise UserError('音声変換モデルの準備が必要です。02_prepare_models.batを実行してください。')
        temporary, _ = await receive_audio(store, file, .5, 60)
        try:
            key = manager.submit('vc', voice_id, source=temporary/'audio.wav', steps=steps)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return {'id': key}

    @app.post('/api/jobs/{key}/cancel')
    def cancel(key: str):
        manager.cancel(key)
        return {'ok': True}

    @app.get('/api/jobs/{key}/file/{extension}')
    def output(key: str, extension: str):
        job = store.get('jobs', key)
        if job['status'] != 'done' or extension not in ('wav', 'mp3'):
            raise UserError('完成した音声だけを保存できます。')
        return FileResponse(store.job_path(key)/f'output.{extension}',
                            media_type='audio/wav' if extension == 'wav' else 'audio/mpeg',
                            filename=f'voiceai-{key[:8]}.{extension}')

    @app.get('/api/jobs/{key}/log')
    def log(key: str):
        store.get('jobs', key)
        path = store.job_path(key)/'worker.log'
        if not path.is_file():
            raise UserError('この処理のログはまだありません。')
        return FileResponse(path, media_type='text/plain', filename=f'voiceai-{key[:8]}.log')

    return app
