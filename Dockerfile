# ベースイメージ: Python 3.11 (軽量版)
FROM python:3.11-slim

# 作業ディレクトリ設定
WORKDIR /app

# 依存関係のインストール (キャッシュ効かせるために先にコピー)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのコピー
COPY src/ src/
COPY data/ data/

# 環境変数の設定 (PYTHONPATHを通す)
ENV PYTHONPATH=/app/src

# 実行コマンド (Uvicornサーバー起動)
# host 0.0.0.0 はコンテナ外からアクセスするために必須
CMD ["uvicorn", "ragqa.server:app", "--host", "0.0.0.0", "--port", "8000"]