外部コンポーネントについて

本配布のアプリ本体には、AIの重み、Python実行環境、FFmpegバイナリ、Seed-VCのソースを同梱していません。セットアップ時に必要な配布元から取得します。以下は主要コンポーネントで、全依存関係の法的な監査結果ではありません。

| コンポーネント | 参照先と条件 |
|---|---|
| Qwen3-TTS | https://github.com/QwenLM/Qwen3-TTS — Apache-2.0 |
| Qwen3-TTS-12Hz-0.6B-Base | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base — Apache-2.0 |
| Seed-VC | https://github.com/Plachtaa/seed-vc — GPL-3.0。取得コミット：51383efd921027683c89e5348211d93ff12ac2a8 |
| Seed-VC重み | https://huggingface.co/Plachta/Seed-VC — 配布ページでGPL-3.0と表示 |
| Seed-VCの関連モデル | CAMPPlus、Whisper、BigVGANなどを上流ローダーから取得。各配布元の条件も適用 |
| Whisper small（参照音声の自動認識） | https://huggingface.co/openai/whisper-small ／ https://github.com/openai/whisper — MIT。既存のTransformers環境を使用 |
| PyTorch／Torchaudio | https://pytorch.org/ — 各配布物のライセンスを参照 |
| uv | https://github.com/astral-sh/uv — MIT／Apache-2.0。0.9.7のZIPを取得しSHA-256を検証 |
| FFmpeg | https://ffmpeg.org/legal.html — imageio-ffmpegの同梱ビルドを使用。ビルド構成によりLGPL／GPL等の条件が適用 |
| FastAPI | https://github.com/fastapi/fastapi — MIT |

ソフトウェアの配布条件と、入力する音声の利用権限は別の事項です。本人または利用許可を得た音声を使用してください。
GPL-3.0のコードを別プロセスにしただけで、配布時の条件がなくなるとは扱いません。将来このアプリを第三者に配布する場合、実際に同梱・結合するコンポーネントと重みに応じて条件を確認してください。
セットアップで得たSeed-VCフォルダー内のLICENSEや著作権表示は保存します。
