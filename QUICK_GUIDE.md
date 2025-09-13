# 🚀 Asia-Pacific Fire Analysis - Quick Guide

**5分で始める火災分析システム** - 初心者でも簡単に大規模火災データ分析を実行できます！

## 📋 必要なもの

✅ **Python 3.8以上**  
✅ **8GB以上のメモリ**  
✅ **インターネット接続**（データダウンロード用）  
✅ **NASA FIRMS APIキー**（無料）

## 🎯 3ステップで完了

### ステップ 1: 環境準備（2分）

```bash
# 1. プロジェクトをダウンロード
git clone https://github.com/yourusername/asia-pacific-fire-analysis.git
cd asia-pacific-fire-analysis

# 2. 仮想環境を作成
python -m venv .venv

# 3. 仮想環境をアクティベート
# Windows PowerShell の場合:
.venv\Scripts\Activate.ps1
# Linux/Mac の場合:
source .venv/bin/activate

# 4. 必要なライブラリをインストール
pip install -r requirements.txt
```

### ステップ 2: APIキー設定（1分）

```bash
# 1. NASA FIRMS APIキーを取得（無料）
# https://firms.modaps.eosdis.nasa.gov/api/ にアクセス
# メールアドレスを入力してAPIキーを取得

# 2. 設定ファイルを編集
# config_asia_pacific_firms.json を開いて以下を変更:
{
  "api_key": "ここにあなたのAPIキーを入力"
}
```

### ステップ 3: 分析実行（2分）

```bash
# システムを実行
python asia_pacific_firms_pipeline.py
```

## 🎉 完了！結果を確認

分析が完了すると、以下のようなフォルダが生成されます：

```
📁 data_firms_YYYYMMDDHHMM/  ← 結果フォルダ
├── 📝 comprehensive_fire_analysis_report.md  ← **ここを見る！**
├── 🖼️ tsne_plot.png                          ← クラスタリング結果
├── 🖼️ cluster_geographic_distribution.png    ← 地図上の分布
├── 🖼️ cluster_temporal_patterns.png          ← 時間パターン
├── 🖼️ cluster_intensity_analysis.png         ← 火災強度分析
└── ...その他のファイル
```

### 📖 レポートの見方

`comprehensive_fire_analysis_report.md` がメインのレポートです：

1. **エグゼクティブサマリー** - 分析結果の要約
2. **地理的分析** - どの地域で火災が多いか
3. **時間パターン** - いつ火災が発生しやすいか
4. **強度分析** - 火災の規模はどの程度か
5. **図表解説** - 6つのグラフの詳細説明

## 💡 よくある質問

### Q: 処理にどのくらい時間がかかりますか？
**A:** 約1-2分程度です。データサイズによって変動しますが、通常は90秒以内に完了します。

### Q: エラーが出ました
**A:** 以下を確認してください：
- インターネット接続
- APIキーが正しく設定されているか
- 十分なメモリ（8GB以上）があるか

### Q: 結果の見方がわかりません
**A:** `comprehensive_fire_analysis_report.md` を開いて、日本語で書かれた詳細な解説をお読みください。

### Q: データを変更したい
**A:** `config_asia_pacific_firms.json` で以下を調整できます：
- `days_back`: データ収集期間（1-10日）
- `max_samples`: 処理するサンプル数
- `confidence_threshold`: 火災検知の信頼度

## 🔧 簡単カスタマイズ

### データ収集期間を変更
```json
{
  "days_back": 7,  ← 7日間のデータを収集（1-10で調整可能）
}
```

### 処理するデータ量を調整
```json
{
  "max_samples": 10000,  ← 1万件まで処理（メモリに応じて調整）
}
```

### 信頼度の高い火災のみ分析
```json
{
  "confidence_threshold": 80,  ← 80%以上の高信頼度のみ（50-100で調整）
}
```

## 🎯 実際の使用例

### 🏢 研究者の場合
「過去1週間のアジア太平洋地域の火災パターンを調べたい」
→ デフォルト設定で実行するだけでOK！

### 🏛️ 政策担当者の場合  
「緊急度の高い火災のみを分析したい」
→ `confidence_threshold: 90` に設定

### 🌍 環境活動家の場合
「詳細な長期トレンドを把握したい」
→ `days_back: 10` で最大期間のデータを収集

## 📞 サポート

### 🆘 問題が発生した場合
1. **エラーメッセージをコピー**
2. **[GitHub Issues](https://github.com/yourusername/asia-pacific-fire-analysis/issues)** に報告
3. **実行環境**（OS、Pythonバージョン）を明記

### 📚 さらに詳しく学びたい場合
- **[技術詳細ドキュメント](README_v1-3_asia.md)** - 上級者向け詳細資料
- **[メインREADME](README.md)** - プロジェクト全体の概要

---

## 🎉 次のステップ

火災分析が成功したら：

1. **結果を共有** - 生成されたレポートや図表を活用
2. **設定を調整** - より具体的な分析ニーズに合わせてカスタマイズ  
3. **定期実行** - 最新の火災トレンドを継続的に監視
4. **コミュニティ参加** - 改善提案やフィードバックを共有

**🌟 すぐに始められる火災分析システムで、重要な洞察を発見しましょう！**

## 🎯 実行例と期待結果

### コマンド実行
```bash
(.venv) PS> python nasa_firms_adaptive_pipeline.py
```

### 期待される出力
```
🚀 Starting NASA FIRMS Adaptive Clustering Pipeline
=== Initializing Pipeline Components ===
✅ GPU memory available: 16.00 GB
✅ Model loaded successfully on cuda

=== Collecting NASA FIRMS Data ===
✅ Generated 64 sample fire detection records

=== Generating Embeddings ===  
✅ Generated 64 embeddings at 220.21 texts/sec

=== Performing Adaptive Clustering ===
✅ HDBSCAN: 2 clusters, quality=0.407
✅ k-means: 3 clusters, quality=0.553
✅ Selection: k-means selected (higher quality)

🎉 Adaptive Pipeline completed successfully!
📊 Quality Score: 0.553 | 🔢 Clusters: 3 | ⏱️ Time: 32.5s
```

## 🛠️ カスタマイズ方法

### NASA FIRMS API使用（実データ）
`config_adaptive_nasa_firms.json`を編集：
```json
{
  "nasa_firms": {
    "map_key": "YOUR_API_KEY_HERE",  // ← NASA FIRMS APIキーを設定
    "area_params": {
      "south": 30.0, "north": 45.0,  // ← 日本周辺（デフォルト）
      "west": 130.0, "east": 145.0
    },
    "days_back": 7                   // ← 過去7日分のデータ
  }
}
```

### クラスタリングパラメータ調整
```json
{
  "adaptive_clustering": {
    "hdbscan_params": {
      "min_cluster_size": 5,    // ← 最小クラスターサイズ
      "min_samples": 3          // ← 最小サンプル数
    },
    "kmeans_params": {
      "n_clusters": 3           // ← k-meansクラスター数
    }
  }
}
```

### 品質評価重み調整
```json
{
  "quality_weights": {
    "silhouette": 0.3,          // ← シルエット係数の重み
    "noise_penalty": 0.2,       // ← ノイズペナルティの重み
    "cluster_balance": 0.1      // ← クラスターバランスの重み
  }
}
```

## 📊 結果の読み方

### 1. コンソール出力
```
🎯 Selected Method: FAISS k-means    // ← 選択された手法
📊 Quality Score: 0.553              // ← 品質スコア（高いほど良い）
🔢 Clusters Found: 3                 // ← 発見されたクラスター数
📉 Noise Ratio: 0.0%                 // ← ノイズ率（HDBSCANのみ）
```

### 2. JSONファイル解読
`final_adaptive_results.json`:
```json
{
  "selected_clustering": {
    "method": "FAISS k-means",        // 選択手法
    "quality_score": 0.553,           // 品質スコア
    "n_clusters": 3,                  // クラスター数
    "processing_time": 0.013          // 処理時間（秒）
  },
  "selection_details": {
    "selection_reason": "k-means selected: higher quality (0.553 vs 0.407)"
  }
}
```

### 3. CSVデータ活用
`nasa_firms_clustered.csv`:
```csv
latitude,longitude,brightness,confidence,cluster_label
41.578,135.015,269.583,89.485,1      // ← cluster_labelが追加
32.054,138.849,285.552,83.049,1
31.701,142.653,270.819,65.618,1
```

## 🔍 トラブルシューティング

### よくある問題と解決法

**Q: GPU使用できない**
```bash
# CPUモードで実行
pip install faiss-cpu  # faiss-gpuの代わり
```

**Q: NASA FIRMS APIエラー**  
→ 自動的にサンプルデータで継続実行されます

**Q: メモリ不足エラー**
```json
// config_adaptive_nasa_firms.jsonで調整
{
  "processing": {
    "max_samples": 100,     // ← サンプル数を減らす
    "batch_size": 50        // ← バッチサイズを減らす
  }
}
```

**Q: 可視化でフォント警告**  
→ 機能に影響なし。日本語フォント不足による警告

## ⚡ 高速実行Tips

### 1. GPU活用
```bash
# GPU版FAISS（高速化）
pip install faiss-gpu
```

### 2. パラメータ最適化
```json
{
  "processing": {
    "batch_size": 100,      // ← GPU性能に応じて調整
    "max_samples": 200      // ← 適度なサンプル数
  }
}
```

### 3. キャッシュ活用
- 埋め込みファイル（embeddings.npy）は再利用可能
- 同じデータなら埋め込み再生成スキップ可能

## 📈 応用例

### 1. 研究用途
```bash
# 複数パラメータでの比較実験
# config調整 → 実行 → 結果保存 → 繰り返し
```

### 2. 運用監視
```bash
# 定期実行でリアルタイム監視
# crontabやタスクスケジューラで自動化
```

### 3. データ探索
```bash
# 異なる地域・期間でのパターン分析
# area_paramsとdays_backを変更して比較
```

## 🎯 期待される活用場面

- **災害対応**: リアルタイム火災検出パターン分析
- **研究**: クラスタリング手法の自動選択システム研究  
- **教育**: 機械学習における手法比較の具体例
- **開発**: 大規模地理データ処理システムのプロトタイプ

---

**実行時間**: 初回 ~3分、2回目以降 ~1分  
**必要環境**: Python 3.8+, 8GB RAM推奨  
**サポート**: エラー時は`pipeline_error.json`をチェック

🚀 **今すぐ試してみましょう！**