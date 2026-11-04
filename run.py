import os
from pathlib import Path
import threading
import webbrowser
import uvicorn

if __name__ == '__main__':
    os.environ.setdefault('VOICEAI_ROOT', str(Path(__file__).resolve().parent))
    from app.main import create_app
    print('VoiceAI Local: http://127.0.0.1:7860')
    print('Close this window or press Ctrl+C to stop.')
    if os.environ.get('VOICEAI_NO_BROWSER') != '1':
        threading.Timer(2, lambda: webbrowser.open('http://127.0.0.1:7860')).start()
    uvicorn.run(create_app(), host='127.0.0.1', port=7860, log_level='info')
