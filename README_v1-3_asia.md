# Asia-Pacific Fire Detection Analysis Pipeline v1.3

## 🌏 概要

このパイプラインは、NASA FIRMS（Fire Information for Resource Management System）データを使用して、アジア太平洋地域の火災検知データに対する大規模なクラスタリング分析と包括的レポート生成を行います。

### 🎯 主要な成果
- **大規模データ処理**: 45,052件→41,734件（高信頼度フィルタ）→15,000件の効率的処理
- **最適化されたクラスタリング**: 3,000件超の大規模データセットに対するFAISS k-means自動選択
- **包括的レポート生成**: 6つの可視化図表を用いた専門的Markdownレポート
- **高品質な分析結果**: クラスタリング品質スコア 0.710、処理時間 84.58秒

## 🏗️ システムアーキテクチャ

### 核心コンポーネント
```
asia_pacific_firms_pipeline.py          # メインパイプライン
├── scripts/data_collector.py           # NASA FIRMSデータ収集
├── scripts/embedding_generator.py      # sentence-transformers埋め込み生成
├── adaptive_clustering_selector.py     # 適応的クラスタリング選択
├── scripts/visualization.py            # t-SNE可視化システム
├── cluster_feature_analyzer.py         # クラスタ特徴分析
└── fire_analysis_report_generator.py   # 包括的レポート生成
```

### 設定ファイル
```
config_asia_pacific_firms.json          # 最適化された設定パラメータ
```

## 🚀 技術仕様

### データ処理能力
- **データソース**: NASA FIRMS VIIRS_SNPP_NRT
- **収集期間**: 最大10日間（API制限内での最適化）
- **対象地域**: アジア太平洋（70°E-180°E, -50°N-50°N）
- **処理規模**: 15,000サンプル（高信頼度フィルタ後）
- **信頼度閾値**: ≥50%（信頼性の高い火災検知のみ）

### 機械学習技術
- **埋め込みモデル**: sentence-transformers/all-MiniLM-L6-v2
- **次元削減**: t-SNE（perplexity=30）
- **クラスタリング**: 適応的選択アルゴリズム
  - **小規模データ（<3,000）**: HDBSCAN（min_cluster_size=50）
  - **大規模データ（≥3,000）**: FAISS k-means（自動選択）
- **GPU加速**: CUDA対応（16GB VRAM利用可能）

### パフォーマンス指標
- **処理速度**: 177.48サンプル/秒
- **埋め込み生成**: 15,000.00テキスト/秒
- **クラスタリング品質**: 0.710（高品質）
- **発見クラスタ数**: 8クラスタ
- **ノイズ率**: 0.0%

## 📊 生成される出力

### 1. データファイル
- `nasa_firms_data.csv` - 収集された生データ（15,000件）
- `embeddings.npy` - 生成された埋め込みベクトル（384次元）
- `asia_pacific_fires_clustered.csv` - クラスタラベル付きデータ
- `final_asia_pacific_results.json` - 分析結果サマリー

### 2. 可視化図表（6つ）
1. **`tsne_plot.png`** - t-SNEクラスタリング可視化
2. **`score_distribution.png`** - スコア分布分析
3. **`cluster_geographic_distribution.png`** - 地理的分布マップ
4. **`cluster_temporal_patterns.png`** - 時間パターン分析
5. **`cluster_intensity_analysis.png`** - 火災強度分析
6. **`cluster_regional_analysis.png`** - 地域特性分析

### 3. 包括的レポート
- **`comprehensive_fire_analysis_report.md`** - 専門的分析レポート
  - エグゼクティブサマリー
  - 技術的手法論の詳細
  - 地理的・時間的・強度分析
  - 6つの図表の詳細解説

## 🔧 実行方法

### 前提条件
```bash
# 仮想環境の準備
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 依存関係インストール
pip install torch sentence-transformers scikit-learn faiss-cpu
pip install pandas numpy matplotlib seaborn plotly
pip install requests langdetect hdbscan umap-learn
```

### 実行手順
```bash
# 1. 設定確認
# config_asia_pacific_firms.json の設定を確認

# 2. パイプライン実行
python asia_pacific_firms_pipeline.py

# 3. 結果確認
# data_firms_YYYYMMDDHHMM/ ディレクトリ内の結果を確認
```

## ⚙️ 設定パラメータ

### 重要な最適化設定（config_asia_pacific_firms.json）
```json
{
  "region": "asia_pacific",
  "days_back": 10,                    # API制限内の最大収集期間
  "max_samples": 15000,               # システム安定性のための制限
  "confidence_threshold": 50,         # 高信頼度フィルタ
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,                 # GPU最適化
    "device": "auto"
  },
  "clustering": {
    "hdbscan_min_cluster_size": 50,   # 小規模データ用
    "kmeans_n_clusters": 15,          # 大規模データ用
    "adaptive_threshold": 3000        # 自動選択閾値
  },
  "visualization": {
    "method": "tsne",                 # t-SNE選択（UMAPではない）
    "perplexity": 30,
    "n_iter": 1000
  }
}
```

## 🎓 実装上の重要な教訓

### 1. スケーラビリティの課題と解決
**課題**: HDBSCANは大規模データ（15,000+サンプル）で処理時間が指数的に増加

**解決策**: 適応的クラスタリング選択アルゴリズムの実装
```python
# adaptive_clustering_selector.py
if len(embeddings) > 3000:
    # 大規模データ：FAISS k-means選択
    method = "faiss_kmeans"
else:
    # 小規模データ：HDBSCAN選択
    method = "hdbscan"
```

### 2. API制限の最適化
**制約**: NASA FIRMS APIの最大10日制限

**最適化**: 
- 30日から10日への期間調整
- 高信頼度フィルタ（≥50%）によるデータ品質向上
- 45,052件→41,734件→15,000件の効率的データ処理

### 3. メモリ管理の最適化
**課題**: 大規模埋め込みデータのメモリ効率

**解決策**:
- バッチサイズ最適化（64サンプル）
- GPU加速活用（16GB VRAM）
- 埋め込み次元の効率化（384次元）

### 4. 可視化システムの安定化
**課題**: t-SNEの収束安定性

**解決策**:
- perplexity=30の最適化
- early_exaggeration期間の調整
- 1000イテレーションでの安定収束

### 5. レポート生成の自動化
**要求**: 6つの図表を用いた包括的分析レポート

**実装**:
- FireAnalysisReportGenerator クラス
- Markdown形式での専門的レポート
- 図表の詳細解説とメタデータ統合

## 📈 パフォーマンス指標

### 処理効率
- **総処理時間**: 84.58秒
- **データ収集**: ~10秒
- **埋め込み生成**: ~5秒（42.51it/s）
- **クラスタリング**: ~15秒
- **可視化**: ~45秒（t-SNE主要部分）
- **レポート生成**: ~5秒

### 品質指標
- **クラスタリング品質**: 0.710（優良）
- **クラスタ数**: 8（適切な分割）
- **ノイズ率**: 0.0%（高品質分離）
- **収束性**: 1000イテレーションで安定

## 🚦 運用上の注意点

### システム要件
- **メモリ**: 最低8GB、推奨16GB
- **GPU**: CUDA対応推奨（高速化）
- **ストレージ**: 実行毎に約500MB

### 制限事項
- NASA FIRMS API: 1日あたりのリクエスト制限あり
- 大規模データ（>15,000）: メモリ不足の可能性
- 処理時間: データサイズに比例して増加

### トラブルシューティング
1. **メモリエラー**: max_samplesを削減
2. **API制限**: days_backを短縮
3. **GPU不使用**: torch.cuda.is_available()を確認
4. **依存関係エラー**: requirements.txtの再インストール

## 🔮 今後の拡張可能性

### 技術的改良
- マルチモーダル分析（衛星画像との統合）
- リアルタイム分析パイプライン
- 分散処理によるスケーラビリティ向上

### 機能拡張
- 他地域への拡張（北米、ヨーロッパ、南米）
- 予測分析機能の追加
- Web API化によるリアルタイムアクセス

## 📋 バージョン履歴

### v1.3 (2024年9月13日)
- 包括的レポート生成機能の追加
- 6つの可視化図表の統合
- 大規模データ最適化の完成
- FireAnalysisReportGenerator実装

### v1.2 
- 適応的クラスタリング選択の実装
- FAISS k-means大規模データ対応
- メモリ最適化とGPU加速

### v1.1
- NASA FIRMS API統合
- t-SNE可視化システム
- 基本的なクラスタリング機能

---

**開発チーム**: DisasterSentiment Project  
**最終更新**: 2024年9月13日  
**ライセンス**: MIT License  
**連絡先**: [プロジェクト詳細]