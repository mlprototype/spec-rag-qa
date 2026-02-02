# 会員登録 設計書（サンプル）

## 構成
- API: POST /api/signup
- DB: users テーブル（email unique）

## バリデーション
- email: RFC準拠までは不要、一般的なメール形式チェック
- password: 8文字以上、英数字混在は将来要件

## 例外設計
- 一意制約違反: 409 Conflict
- バリデーションエラー: 400 Bad Request
