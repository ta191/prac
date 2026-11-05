import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import wave
from array import array
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import create_app
from app.core import Store, UserError, ffmpeg, worker_environment


def sample(seconds=3.2, silent=False):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        values = array('h', (0 if silent else int(5000*math.sin(2*math.pi*220*i/24000)) for i in range(int(seconds*24000))))
        if sys.byteorder != 'little': values.byteswap()
        wav.writeframes(values.tobytes())
    return buf.getvalue()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []
        self.app = create_app(self.root, self.fixture_runner)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.headers = {'X-VoiceAI-Token': self.app.state.token}
        self.sound = sample()
        self.ready = patch('app.main.runtime_status', return_value={'tts':True,'vc':True,'audio':True})
        self.ready.start()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.ready.stop()
        self.temp.cleanup()

    def fixture_runner(self, job, manager):
        # Test fixture only. Production always invokes the actual AI subprocess.
        folder = manager.store.job_path(job['id'])
        request = json.loads((folder/'request.json').read_text('utf-8'))
        self.calls.append(request)
        shutil.copyfile(request['reference'], folder/'output.wav')
        subprocess.run([ffmpeg(self.root), '-nostdin', '-v', 'error', '-y', '-i', str(folder/'output.wav'),
                        str(folder/'output.mp3')], check=True, capture_output=True)

    def voice(self, name='テスト用の声', audio=None, transcript='こんにちは。'):
        return self.client.post('/api/voices', headers=self.headers,
                                data={'name': name, 'transcript':transcript, 'consent':'true'},
                                files={'file':('reference.wav', self.sound if audio is None else audio, 'audio/wav')})

    def wait(self, key, status='done'):
        deadline = time.monotonic()+6
        while time.monotonic()<deadline:
            job = next(j for j in self.client.get('/api/jobs').json() if j['id']==key)
            if job['status'] == status: return job
            if job['status'] in ('failed','cancelled') and status != job['status']:
                self.fail(str(job))
            time.sleep(.03)
        self.fail('Timed out waiting for '+status)

    def tts(self, key, text='こんにちは。今日は2026年11月3日です。'):
        return self.client.post('/api/jobs/tts', headers=self.headers, json={'voice_id':key,'text':text})

    def test_page_and_assets(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code,200)
        self.assertIn('こえのアトリエ',page.text)
        self.assertIn(self.app.state.token,page.text)
        self.assertIn("frame-ancestors 'none'",page.headers['content-security-policy'])
        for asset in ('app.js','style.css'):
            self.assertEqual(self.client.get('/assets/'+asset).status_code,200)

    def test_csrf_and_foreign_origins(self):
        payload={'voice_id':str(uuid.uuid4()),'text':'こんにちは'}
        self.assertEqual(self.client.post('/api/jobs/tts',json=payload).status_code,403)
        self.assertEqual(self.client.post('/api/jobs/tts',json=payload,headers={**self.headers,'Origin':'https://example.com'}).status_code,403)
        self.assertEqual(self.client.get('/api/voices',headers={'Host':'attacker.example'}).status_code,400)

    def test_registration_shared_voice_and_real_audio_exports(self):
        response=self.voice();self.assertEqual(response.status_code,200,response.text)
        key=response.json()['id']
        tts=self.tts(key).json()['id']; self.wait(tts)
        vc=self.client.post('/api/jobs/vc',headers=self.headers,data={'voice_id':key,'steps':25},files={'file':('source.wav',self.sound,'audio/wav')})
        self.assertEqual(vc.status_code,200,vc.text);self.wait(vc.json()['id'])
        self.assertEqual(self.calls[0]['reference'],self.calls[1]['reference'])
        self.assertEqual(self.calls[0]['transcript'],'')
        self.assertEqual(self.calls[0]['transcript_source'],'automatic_pending')
        for extension in ('wav','mp3'):
            exported=self.client.get(f'/api/jobs/{tts}/file/{extension}')
            self.assertEqual(exported.status_code,200)
            self.assertGreater(len(exported.content),100)
            path=self.root/f'export.{extension}';path.write_bytes(exported.content)
            subprocess.run([ffmpeg(self.root),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)
        self.assertEqual(list((self.root/'data'/'tmp').iterdir()),[])

    def test_persistence_rename_delete(self):
        key=self.voice().json()['id']
        response=self.client.put(f'/api/voices/{key}',headers=self.headers,json={'name':'新しい名前','transcript':'おはようございます。'})
        self.assertEqual(response.status_code,200)
        store=Store(self.root)
        self.assertEqual(store.get('voices',key)['name'],'新しい名前')
        self.assertTrue(store.voice_path(key).is_file())
        self.assertEqual(self.client.delete(f'/api/voices/{key}',headers=self.headers).status_code,200)
        self.assertFalse(store.voice_path(key).exists())
        self.assertEqual(self.client.get('/api/voices').json(),[])

    def test_bad_audio_silence_duration_and_empty_name(self):
        for audio in (b'not audio',sample(silent=True),sample(.2),sample(31)):
            result=self.voice(audio=audio)
            self.assertEqual(result.status_code,400,result.text)
        self.assertEqual(self.voice(name='  ').status_code,400)
        self.assertEqual(self.client.get('/api/voices').json(),[])
        self.assertEqual(list((self.root/'data'/'tmp').iterdir()),[])

    def test_consent_and_format_required(self):
        for consent,filename in [('false','voice.wav'),('true','voice.html')]:
            result=self.client.post('/api/voices',headers=self.headers,data={'name':'声','consent':consent},files={'file':(filename,self.sound)})
            self.assertEqual(result.status_code,400,result.text)

    def test_tts_limits_and_unknown_ids(self):
        key=self.voice().json()['id']
        self.assertEqual(self.tts(key,' '*4).status_code,400)
        self.assertEqual(self.tts(key,'あ'*501).status_code,422)
        self.assertEqual(self.tts('not-a-uuid').status_code,400)
        self.assertEqual(self.client.get('/api/jobs/not-a-uuid/file/wav').status_code,400)

    def test_models_not_prepared(self):
        key=self.voice().json()['id']
        with patch('app.main.runtime_status',return_value={'tts':False,'vc':False,'audio':True}):
            response=self.tts(key)
            self.assertEqual(response.status_code,400)
            self.assertIn('START.bat',response.text)

    def test_failure_then_next_job_runs(self):
        key=self.voice().json()['id']
        def fail(job,manager):raise UserError('テスト用エラー')
        self.app.state.manager.runner=fail
        job=self.tts(key).json()['id'];self.wait(job,'failed')
        self.app.state.manager.runner=self.fixture_runner
        self.wait(self.tts(key).json()['id'])

    def test_serial_queue_cancel_and_protect_active_voice(self):
        key=self.voice().json()['id']
        gate=threading.Event();entered=threading.Event();calls=[]
        def blocked(job,manager):
            calls.append(job['id']);entered.set()
            while not gate.wait(.03):
                if manager.interrupted(job['id']):return
            self.fixture_runner(job,manager)
        self.app.state.manager.runner=blocked
        first=self.tts(key).json()['id'];self.assertTrue(entered.wait(3))
        second=self.tts(key).json()['id']
        self.assertEqual(self.client.delete(f'/api/voices/{key}',headers=self.headers).status_code,400)
        self.assertEqual(self.client.post(f'/api/jobs/{second}/cancel',headers=self.headers).status_code,200)
        gate.set();self.wait(first);self.wait(second,'cancelled')
        self.assertEqual(calls,[first])
        self.assertEqual(self.client.delete(f'/api/voices/{key}',headers=self.headers).status_code,200)

    def test_running_subprocess_cancellation(self):
        key=self.voice().json()['id'];entered=threading.Event()
        def runner(job,manager):
            entered.set()
            manager.run_process([sys.executable,'-c','import time; time.sleep(30)'],job['id'],manager.store.job_path(job['id'])/'worker.log',worker_environment(self.root),40)
        self.app.state.manager.runner=runner
        job=self.tts(key).json()['id'];self.assertTrue(entered.wait(3))
        self.client.post(f'/api/jobs/{job}/cancel',headers=self.headers);self.wait(job,'cancelled')
        deadline=time.monotonic()+3
        while job in self.app.state.manager.cancelled and time.monotonic()<deadline:time.sleep(.03)
        self.assertNotIn(job,self.app.state.manager.cancelled)
        self.assertEqual(self.client.get(f'/api/jobs/{job}/file/wav').status_code,400)

    def test_restart_marks_interrupted_jobs(self):
        store=self.app.state.store
        key=str(uuid.uuid4())
        with store.connect() as con:
            con.execute("INSERT INTO jobs (id,kind,voice_id,voice_name,text,status,message,created) VALUES (?,?,?,?,?,?,?,?)",(key,'tts',str(uuid.uuid4()),'声','文','running','処理中','2026-11-03'))
        restarted=Store(self.root)
        self.assertEqual(restarted.get('jobs',key)['status'],'failed')

    def test_environment_offline_and_local_cache(self):
        env=worker_environment(self.root)
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
        self.assertTrue(env['HF_HOME'].startswith(str(self.root)))

    def test_registration_without_transcript_and_name_only_edit(self):
        response = self.client.post('/api/voices', headers=self.headers,
                                   data={'name':'音声のみ', 'consent':'true'},
                                   files={'file':('reference.wav', self.sound, 'audio/wav')})
        self.assertEqual(response.status_code, 200, response.text)
        key = response.json()['id']
        self.assertEqual(response.json()['transcript'], '')
        updated = self.client.put(f'/api/voices/{key}', headers=self.headers, json={'name':'名前変更'})
        self.assertEqual(updated.status_code, 200, updated.text)
        self.wait(self.tts(key, '今日天気いいね。').json()['id'])
        self.assertEqual(self.calls[-1]['reference'], str(self.app.state.store.voice_path(key)))
        self.assertNotIn('voice-transcript', self.client.get('/').text)

    def test_real_runner_orders_asr_before_tts_and_ignores_manual_text(self):
        from app.core import runtime_paths
        for path in runtime_paths(self.root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        manager = self.app.state.manager
        stages = []
        requests = []
        real_process = manager.run_process
        def process(command, key, logpath, env, timeout):
            if len(command) > 2 and command[2] in ('asr', 'tts'):
                kind = command[2]
                stages.append(kind)
                folder = manager.store.job_path(key)
                request = json.loads((folder/'request.json').read_text('utf-8'))
                requests.append(request)
                self.assertEqual(env['HF_HUB_OFFLINE'], '1')
                if kind == 'asr':
                    prepared = Path(request['reference']).with_name('tts-reference.wav')
                    shutil.copyfile(request['reference'], prepared)
                    (folder/'reference-analysis.json').write_text(json.dumps({'text':'自動認識した全文。', 'reference':str(prepared)}), 'utf-8')
                else:
                    shutil.copyfile(request['reference'], folder/'output.wav')
            else:
                real_process(command, key, logpath, env, timeout)
        manager.runner = manager.run_worker
        with patch.object(manager, 'run_process', side_effect=process):
            key = self.voice(transcript='途中までの手入力').json()['id']
            self.wait(self.tts(key).json()['id'])
        self.assertEqual(stages, ['asr', 'tts'])
        self.assertEqual(requests[0]['transcript'], '')
        self.assertEqual(requests[1]['transcript'], '自動認識した全文。')
        self.assertEqual(requests[1]['transcript_source'], 'automatic')
        self.assertEqual(Path(requests[0]['reference']).parent, Path(requests[1]['reference']).parent)
        self.assertEqual(Path(requests[1]['reference']).name, 'tts-reference.wav')

    def test_cancel_during_asr_does_not_start_tts(self):
        from app.core import runtime_paths
        for path in runtime_paths(self.root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        manager = self.app.state.manager
        stages = []
        def process(command, key, *args, **kwargs):
            stages.append(command[2])
            manager.cancel(key)
        manager.runner = manager.run_worker
        with patch.object(manager, 'run_process', side_effect=process):
            key = self.voice().json()['id']
            self.wait(self.tts(key).json()['id'], 'cancelled')
        self.assertEqual(stages, ['asr'])


if __name__=='__main__':unittest.main()
