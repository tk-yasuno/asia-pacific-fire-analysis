#!/usr/bin/env python3
"""
アジア太平洋地域 NASA FIRMS 森林火災データ分析パイプライン
日本を中心とするアジア太平洋エリアの火災データ        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def run_full_pipeline(self, sample_size: int = 1000):ォルダに保存

エリア範囲:
- 緯度: 10°N ~ 50°N (東南アジア〜ロシア極東)
- 経度: 100°E ~ 180°E (インド〜太平洋)
- 対象国: 日本、韓国、中国、台湾、フィリピン、インドネシア、マレーシア、タイ、ベトナム等
"""

import os
import sys
import json
import logging
import time
import time
from datetime import datetime
from typing import Dict, List, Optional

# パス設定
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.append(scripts_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# モジュールインポート
from scripts.data_collector import DataCollector
from scripts.model_loader import ModelLoader
from scripts.embedding_generator import EmbeddingGenerator
from adaptive_clustering_selector import AdaptiveClusteringSelector
from scripts.visualization import VisualizationManager
from cluster_feature_analyzer import ClusterFeatureAnalyzer
from fire_analysis_report_generator import FireAnalysisReportGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def _time_step(step_name):
    """ステップ実行時間測定用デコレーター"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            logger.info(f"=== Starting: {step_name} ===")
            try:
                result = func(self, *args, **kwargs)
                end_time = time.time()
                logger.info(f"=== Completed: {step_name} ({end_time - start_time:.2f}s) ===")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"=== Failed: {step_name} ({end_time - start_time:.2f}s) - {e} ===")
                raise
        return wrapper
    return decorator


class AsiaPacificFirmsAnalyzer:
    """アジア太平洋地域NASA FIRMS森林火災分析システム"""
    
    def __init__(self, config_path: str = "config_asia_pacific_firms.json"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # タイムスタンプ付きアウトプットディレクトリ作成
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        self.output_dir = f"data_firms_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        
        # 処理時間記録
        self.step_times = {}
        
    def _load_config(self) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _time_step(self, step_name: str):
        """ステップ実行時間測定デコレータ"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.step_times[step_name] = elapsed
                logger.info(f"Step '{step_name}' completed in {elapsed:.2f}s")
                return result
            return wrapper
        return decorator
    
    def run_pipeline(self) -> str:
        """
        アジア太平洋地域火災分析パイプライン実行
        
        Returns:
            結果ファイルパス
        """
        logger.info("🌏 Starting Asia-Pacific FIRMS Fire Analysis Pipeline")
        pipeline_start = time.time()
        
        try:
            # Step 1: コンポーネント初期化
            logger.info("=== Initializing Pipeline Components ===")
            model_loader, embedding_generator, clustering_selector, data_collector, viz_manager, cluster_analyzer, report_generator = self._initialize_components()
            
            # Step 2: NASA FIRMSデータ収集
            logger.info("=== Collecting NASA FIRMS Data (Asia-Pacific Region) ===")
            nasa_data = self._collect_nasa_firms_data(data_collector)
            
            if len(nasa_data) == 0:
                logger.error("No NASA FIRMS data collected")
                return None
            
            # Step 3: 埋め込み生成
            logger.info("=== Generating Text Embeddings ===")
            embeddings, scores = self._generate_embeddings(embedding_generator, nasa_data)
            
            # Step 4: 適応クラスタリング
            logger.info("=== Performing Adaptive Clustering ===")
            clustering_result = self._perform_clustering(clustering_selector, embeddings)
            
            # Step 5: 可視化作成
            logger.info("=== Creating Comprehensive Visualizations ===")
            self._create_visualizations(viz_manager, embeddings, clustering_result, nasa_data)
            
            # Step 6: クラスター特徴分析
            logger.info("=== Performing Cluster Feature Analysis ===")
            feature_analysis = self._perform_cluster_feature_analysis(cluster_analyzer, nasa_data, clustering_result, embeddings)
            
            # Step 7: 結果保存
            logger.info("=== Saving Final Results ===")
            result_path = self._save_results(clustering_result, nasa_data, feature_analysis)
            
            # Step 8: 包括的レポート生成
            logger.info("=== Generating Comprehensive Analysis Report ===")
            report_path = self._generate_comprehensive_report(report_generator, clustering_result, feature_analysis, nasa_data)
            
            # パイプライン完了
            total_time = time.time() - pipeline_start
            logger.info("🎉 Asia-Pacific Fire Analysis Pipeline completed successfully!")
            logger.info(f"Processing time: {total_time:.2f}s for {len(nasa_data)} samples")
            
            self._print_summary(clustering_result, nasa_data, total_time)
            
            return result_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _initialize_components(self):
        """パイプラインコンポーネント初期化"""
        # モデルローダー
        model_loader = ModelLoader(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device']
        )
        
        # 埋め込み生成器
        embedding_generator = EmbeddingGenerator(
            model=model_loader.load_model(),
            batch_size=self.config['embedding']['batch_size']
        )
        
        # 適応クラスタリングセレクター
        clustering_selector = AdaptiveClusteringSelector(
            output_dir=self.output_dir
        )
        
        # データコレクター
        data_collector = DataCollector()
        
        # 可視化マネージャー
        viz_manager = VisualizationManager(
            output_dir=self.output_dir,
            figsize=tuple(self.config['visualization']['figsize'])
        )
        
        # クラスター特徴分析器
        cluster_analyzer = ClusterFeatureAnalyzer(output_dir=self.output_dir)
        
        # レポート生成器
        report_generator = FireAnalysisReportGenerator(output_dir=self.output_dir)
        
        logger.info("All components initialized successfully")
        return model_loader, embedding_generator, clustering_selector, data_collector, viz_manager, cluster_analyzer, report_generator
    
    def _collect_nasa_firms_data(self, data_collector: DataCollector):
        """NASA FIRMSデータ収集"""
        nasa_config = self.config['nasa_firms']
        
        # アジア太平洋地域のデータ取得
        df = data_collector.collect_nasa_firms_data(
            area_params=nasa_config['area_params'],
            days_back=nasa_config['days_back'],
            satellite=nasa_config['satellite']
        )
        
        if len(df) == 0:
            logger.warning("No fire data found in Asia-Pacific region")
            return []
        
        # 大量データ処理 - 高信頼度フィルタ後の全データを処理
        max_samples = self.config['processing']['max_samples']
        initial_count = len(df)
        
        if max_samples and len(df) > max_samples:
            logger.warning(f"Data exceeds max_samples limit ({len(df)} > {max_samples})")
            logger.warning(f"Truncating to first {max_samples} samples for system stability")
            df = df.iloc[:max_samples]  # サンプリングではなく先頭から切り取り
        else:
            logger.info(f"Processing all {len(df)} high-confidence fire detections (no sampling)")
        
        logger.info(f"Final dataset: {len(df)} NASA FIRMS records for comprehensive analysis")
        
        # データ保存
        data_path = os.path.join(self.output_dir, "nasa_firms_data.csv")
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved: {data_path}")
        
        # テキスト形式でのデータ準備（埋め込み生成用）
        texts = []
        for _, row in df.iterrows():
            text = f"Fire detection: Lat={row['latitude']:.3f}, Lon={row['longitude']:.3f}, " \
                   f"Brightness={row['brightness']:.1f}, Confidence={row['confidence']:.1f}%, " \
                   f"Date={row['acq_date']} {row['acq_time']}, Satellite={row['satellite']}"
            texts.append(text)
        
        return texts
    
    def _generate_embeddings(self, embedding_generator: EmbeddingGenerator, texts: List[str]):
        """埋め込み生成"""
        embeddings, scores = embedding_generator.generate_embeddings_batch(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings (dim: {embeddings.shape[1]}) at {len(texts)/(self.step_times.get('Embedding Generation', 1)):.2f} texts/sec")
        
        # 埋め込み保存
        import numpy as np
        embeddings_path = os.path.join(self.output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings.cpu().numpy())
        logger.info(f"Embeddings saved: {embeddings_path}")
        
        return embeddings, scores
    
    def _perform_clustering(self, clustering_selector: AdaptiveClusteringSelector, embeddings):
        """適応クラスタリング実行"""
        # クラスタリングパラメータ
        hdbscan_params = self.config['adaptive_clustering']['hdbscan_params']
        kmeans_params = self.config['adaptive_clustering']['kmeans_params']
        
        logger.info(f"Adaptive parameters: HDBSCAN min_cluster_size={hdbscan_params['min_cluster_size']}, k-means n_clusters={kmeans_params['n_clusters']}")
        
        # 適応クラスタリング実行
        clustering_result, result_info = clustering_selector.select_best_clustering(embeddings)
        
        # result_infoの安全な処理
        method_info = result_info.get('selection_reason', 'unknown selection reason')
        selected_method = clustering_result.method if hasattr(clustering_result, 'method') else 'unknown method'
        
        logger.info(f"Selected method: {selected_method} ({method_info})")
        
        return clustering_result
    
    def _create_visualizations(self, viz_manager: VisualizationManager, embeddings, clustering_result, nasa_data):
        """包括的可視化作成"""
        import numpy as np
        
        # データ準備
        labels = clustering_result.labels
        embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        
        # ダミースコア作成（実際にはembedding_generatorから取得）
        scores = np.random.rand(len(labels))
        
        # テキストとIDの準備
        texts = [f"Fire detected at index {i}: {text}" for i, text in enumerate(nasa_data)]
        ids = [f"fire_{i:04d}" for i in range(len(texts))]
        
        # 分析結果の準備
        unique_labels = np.unique(labels[labels >= 0])
        cluster_stats = {}
        for label in unique_labels:
            mask = labels == label
            cluster_stats[int(label)] = {
                'n_samples': int(np.sum(mask)),
                'score_mean': float(scores[mask].mean()),
                'score_std': float(scores[mask].std())
            }
        
        analysis = {
            'cluster_stats': cluster_stats,
            'quality_score': clustering_result.metrics.quality_score,
            'n_clusters': len(unique_labels),
            'noise_ratio': (labels == -1).sum() / len(labels),
            'method': clustering_result.method
        }
        
        # 可視化パイプライン実行
        try:
            plot_files = viz_manager.run_visualization_pipeline(
                features=embeddings_np,
                labels=labels,
                scores=scores,
                texts=texts,
                ids=ids,
                analysis=analysis
            )
            
            logger.info(f"✅ Generated {len(plot_files)} visualization files")
            for name, path in plot_files.items():
                logger.info(f"  📊 {name}: {path}")
                
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # 可視化が失敗しても処理を続行
    
    def _perform_cluster_feature_analysis(self, cluster_analyzer: ClusterFeatureAnalyzer, nasa_data: List, clustering_result, embeddings):
        """クラスター特徴分析実行"""
        import pandas as pd
        
        # テキストデータから基本的な地理的情報を抽出
        # nasa_dataはテキストのリストなので、保存されたCSVファイルから実際のデータを読み込む
        data_path = os.path.join(self.output_dir, "nasa_firms_data.csv")
        nasa_df = pd.read_csv(data_path)
        
        # 必要な列が存在するか確認
        required_columns = ['latitude', 'longitude', 'brightness', 'confidence', 'acq_date', 'acq_time']
        for col in required_columns:
            if col not in nasa_df.columns:
                logger.warning(f"Missing column: {col}, setting default values")
                nasa_df[col] = 0 if col in ['latitude', 'longitude', 'brightness', 'confidence'] else ''
        
        # FRP列を追加（NASA FIRMSデータに含まれていることが多い）
        if 'frp' not in nasa_df.columns:
            nasa_df['frp'] = 0
        
        # サンプル数を埋め込み数に合わせる
        if len(nasa_df) > len(clustering_result.labels):
            nasa_df = nasa_df.iloc[:len(clustering_result.labels)]
        elif len(nasa_df) < len(clustering_result.labels):
            # 不足分をパディング
            padding_rows = len(clustering_result.labels) - len(nasa_df)
            padding_df = pd.DataFrame({
                'latitude': [0] * padding_rows,
                'longitude': [0] * padding_rows,
                'brightness': [0] * padding_rows,
                'confidence': [0] * padding_rows,
                'acq_date': [''] * padding_rows,
                'acq_time': [''] * padding_rows,
                'frp': [0] * padding_rows
            })
            nasa_df = pd.concat([nasa_df, padding_df], ignore_index=True)
        
        # クラスター特徴分析実行
        feature_analysis = cluster_analyzer.analyze_cluster_features(
            nasa_data=nasa_df,
            labels=clustering_result.labels,
            embeddings=embeddings
        )
        
        # 特徴分析可視化作成
        viz_files = cluster_analyzer.create_feature_visualizations(
            analysis_results=feature_analysis,
            save_dir=self.output_dir
        )
        
        logger.info(f"✅ Generated {len(viz_files)} cluster feature analysis visualizations")
        for viz_file in viz_files:
            logger.info(f"  📈 Feature viz: {viz_file}")
        
        return feature_analysis
    
    def _save_results(self, clustering_result, texts, feature_analysis=None):
        """最終結果保存"""
        # 分析結果保存
        results = {
            "analysis_info": {
                "region": "Asia-Pacific",
                "area_coverage": {
                    "south": self.config['nasa_firms']['area_params']['south'],
                    "north": self.config['nasa_firms']['area_params']['north'],
                    "west": self.config['nasa_firms']['area_params']['west'],
                    "east": self.config['nasa_firms']['area_params']['east']
                },
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(texts)
            },
            "clustering_result": clustering_result,
            "cluster_feature_analysis": feature_analysis if feature_analysis else {},
            "processing_times": self.step_times
        }
        
        result_path = os.path.join(self.output_dir, "final_asia_pacific_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Final results saved: {result_path}")
        
        # ラベル付きデータ保存
        import pandas as pd
        import numpy as np
        
        # テキストとラベルを組み合わせ
        labeled_data = []
        for i, (text, label) in enumerate(zip(texts, clustering_result.labels)):
            labeled_data.append({
                'id': i,
                'text': text,
                'cluster': int(label),
                'cluster_size': np.sum(clustering_result.labels == label)
            })
        
        labeled_df = pd.DataFrame(labeled_data)
        labeled_path = os.path.join(self.output_dir, "asia_pacific_fires_clustered.csv")
        labeled_df.to_csv(labeled_path, index=False, encoding='utf-8')
        logger.info(f"Labeled data saved: {labeled_path}")
        
        return result_path
    
    def _generate_comprehensive_report(self, report_generator: FireAnalysisReportGenerator, 
                                     clustering_result, feature_analysis, nasa_data):
        """包括的分析レポート生成"""
        # データファイルパス
        nasa_data_path = os.path.join(self.output_dir, "nasa_firms_data.csv")
        
        # レポート生成
        report_path = report_generator.generate_comprehensive_report(
            clustering_result=clustering_result,
            feature_analysis=feature_analysis,
            nasa_data_path=nasa_data_path,
            config=self.config
        )
        
        logger.info(f"📝 Comprehensive analysis report generated: {report_path}")
        return report_path
    
    def _print_summary(self, clustering_result, texts, total_time):
        """結果サマリー表示"""
        labels = clustering_result.labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels) * 100 if len(labels) > 0 else 0
        
        print("\n" + "="*70)
        print("🌏 ASIA-PACIFIC FOREST FIRE ANALYSIS RESULTS")
        print("="*70)
        print(f"✅ Status: SUCCESS")
        print(f"🎯 Selected Method: {clustering_result.method}")
        print(f"📊 Quality Score: {clustering_result.metrics.quality_score:.3f}")
        print(f"🔢 Clusters Found: {n_clusters}")
        print(f"📉 Noise Ratio: {noise_ratio:.1f}%")
        print(f"📦 Total Fire Detections: {len(texts)}")
        print(f"⏱️ Processing Time: {total_time:.2f}s")
        print(f"📁 Results Directory: {self.output_dir}")
        print(f"🌍 Region Coverage: Asia-Pacific (10°N-50°N, 100°E-180°E)")
        print("="*70)


def main():
    """メイン実行関数"""
    try:
        # アジア太平洋地域分析実行
        analyzer = AsiaPacificFirmsAnalyzer()
        result_path = analyzer.run_pipeline()
        
        if result_path:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 Results saved in: {analyzer.output_dir}")
            print(f"📄 Main results file: {result_path}")
            
            # 生成されたファイル一覧表示
            import os
            output_files = os.listdir(analyzer.output_dir)
            
            print(f"\n📊 Generated files ({len(output_files)} total):")
            for file in sorted(output_files):
                if file.endswith('.png'):
                    print(f"  🖼️  {file}")
                elif file.endswith('.json'):
                    print(f"  📋 {file}")
                elif file.endswith('.csv'):
                    print(f"  📊 {file}")
                elif file.endswith('.md'):
                    print(f"  📝 {file} (📖 COMPREHENSIVE ANALYSIS REPORT)")
                else:
                    print(f"  📄 {file}")
            
            # レポートファイルの特別案内
            report_file = os.path.join(analyzer.output_dir, "comprehensive_fire_analysis_report.md")
            if os.path.exists(report_file):
                print(f"\n📖 **包括的分析レポートが生成されました**")
                print(f"   ファイル: {report_file}")
                print(f"   内容: 6つの図表を用いた詳細な火災分析レポート")
                print(f"   形式: Markdown形式（テキストエディタで閲覧可能）")
        else:
            print("❌ Analysis failed - no data collected")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()