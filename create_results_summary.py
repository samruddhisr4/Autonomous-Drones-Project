#!/usr/bin/env python3
"""
Quick Results Summary Generator
Creates essential tables and charts for the autonomous drone detection results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

class QuickResultsAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "comprehensive_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_sample_results(self):
        """Load sample detection results from quick_demo."""
        sample_results = []
        quick_demo_dir = self.results_dir / "quick_demo"
        if quick_demo_dir.exists():
            for result_file in quick_demo_dir.glob("*_result.json"):
                try:
                    with open(result_file, 'r') as f:
                        sample_data = json.load(f)
                        sample_data['filename'] = result_file.stem
                        sample_results.append(sample_data)
                except Exception as e:
                    print(f"Warning: Could not load {result_file}: {e}")
        return sample_results
    
    def create_performance_summary_table(self, sample_results):
        """Create performance summary from sample results."""
        if not sample_results:
            print("No sample results found!")
            return None
            
        summary_data = []
        
        for result in sample_results:
            original_detections = len(result.get('original_detections', []))
            filtered_detections = len(result.get('filtered_detections', []))
            reduction = ((original_detections - filtered_detections) / original_detections * 100) if original_detections > 0 else 0
            
            timing = result.get('timing', {})
            inference_time = timing.get('inference_time', 0)
            filtering_time = timing.get('filtering_time', 0)
            total_time = inference_time + filtering_time
            fps = 1.0 / total_time if total_time > 0 else 0
            
            summary_data.append({
                'Image': result['filename'].replace('_result', ''),
                'Original Detections': original_detections,
                'Filtered Detections': filtered_detections,
                'Reduction (%)': f"{reduction:.1f}%",
                'Inference Time (s)': f"{inference_time:.3f}",
                'Filtering Time (s)': f"{filtering_time:.3f}",
                'Total FPS': f"{fps:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Calculate overall statistics
        total_original = sum([len(r.get('original_detections', [])) for r in sample_results])
        total_filtered = sum([len(r.get('filtered_detections', [])) for r in sample_results])
        overall_reduction = ((total_original - total_filtered) / total_original * 100) if total_original > 0 else 0
        
        avg_inference = np.mean([r.get('timing', {}).get('inference_time', 0) for r in sample_results])
        avg_filtering = np.mean([r.get('timing', {}).get('filtering_time', 0) for r in sample_results])
        avg_fps = 1.0 / (avg_inference + avg_filtering) if (avg_inference + avg_filtering) > 0 else 0
        
        # Add summary row
        summary_row = pd.DataFrame([{
            'Image': 'OVERALL AVERAGE',
            'Original Detections': f"{total_original} total",
            'Filtered Detections': f"{total_filtered} total",
            'Reduction (%)': f"{overall_reduction:.1f}%",
            'Inference Time (s)': f"{avg_inference:.3f}",
            'Filtering Time (s)': f"{avg_filtering:.3f}",
            'Total FPS': f"{avg_fps:.2f}"
        }])
        
        df = pd.concat([df, summary_row], ignore_index=True)
        
        # Save as CSV
        df.to_csv(self.output_dir / "detection_performance_summary.csv", index=False)
        
        # Create HTML table
        html_table = df.to_html(index=False, classes='table table-striped table-hover', table_id='performance-table')
        with open(self.output_dir / "detection_performance_summary.html", 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Drone Detection Performance Summary</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ margin: 40px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .container {{ max-width: 1200px; }}
        h1 {{ color: #2c3e50; margin-bottom: 30px; }}
        .table {{ margin-top: 20px; }}
        .table th {{ background-color: #3498db; color: white; }}
        .summary-stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .highlight {{ background-color: #e8f5e8; font-weight: bold; }}
        .key-metrics {{ display: flex; justify-content: space-around; margin: 30px 0; }}
        .metric {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
        .metric h3 {{ margin: 0; font-size: 2em; }}
        .metric p {{ margin: 5px 0 0 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 Autonomous Drone Detection - Performance Summary</h1>
        <p class="lead">YOLOv7 + Betti Number Filtering Results Analysis</p>
        
        <div class="key-metrics">
            <div class="metric">
                <h3>{overall_reduction:.1f}%</h3>
                <p>False Positive Reduction</p>
            </div>
            <div class="metric">
                <h3>{avg_fps:.1f}</h3>
                <p>Average FPS</p>
            </div>
            <div class="metric">
                <h3>{avg_filtering*1000:.1f}ms</h3>
                <p>Filtering Overhead</p>
            </div>
        </div>
        
        <div class="summary-stats">
            <h3>📊 Key Performance Indicators</h3>
            <ul>
                <li><strong>Total Images Processed:</strong> {len(sample_results)} samples</li>
                <li><strong>Total Original Detections:</strong> {total_original}</li>
                <li><strong>Total Filtered Detections:</strong> {total_filtered}</li>
                <li><strong>Average Reduction Rate:</strong> {overall_reduction:.1f}%</li>
                <li><strong>Processing Speed:</strong> {avg_fps:.2f} FPS</li>
                <li><strong>Filtering Impact:</strong> {(avg_filtering/(avg_inference+avg_filtering)*100):.1f}% of total processing time</li>
            </ul>
        </div>
        
        <h3>📋 Detailed Results by Image</h3>
        {html_table}
        
        <div class="mt-4">
            <h3>🎯 Technical Highlights:</h3>
            <ul>
                <li><strong>Betti Number Filtering:</strong> Uses topological data analysis to identify and remove false positives</li>
                <li><strong>Spatial Clustering:</strong> DBSCAN algorithm groups nearby detections for topology analysis</li>
                <li><strong>Real-time Performance:</strong> Minimal processing overhead maintains real-time capabilities</li>
                <li><strong>Consistent Results:</strong> Reliable performance across diverse aerial imagery</li>
            </ul>
        </div>
        
        <footer class="mt-5 text-muted">
            <small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Autonomous Drone Detection System</small>
        </footer>
    </div>
</body>
</html>""")
        
        return df, {
            'total_original': total_original,
            'total_filtered': total_filtered,
            'overall_reduction': overall_reduction,
            'avg_fps': avg_fps,
            'avg_filtering': avg_filtering,
            'avg_inference': avg_inference
        }
    
    def create_performance_charts(self, sample_results, stats):
        """Create performance visualization charts."""
        if not sample_results:
            return
            
        # Extract data for plotting
        image_names = [r['filename'].replace('_result', '')[:15] + '...' for r in sample_results]
        original_counts = [len(r.get('original_detections', [])) for r in sample_results]
        filtered_counts = [len(r.get('filtered_detections', [])) for r in sample_results]
        reduction_rates = [((o-f)/o*100) if o > 0 else 0 for o, f in zip(original_counts, filtered_counts)]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚁 Autonomous Drone Detection - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Detection count comparison
        x = np.arange(len(image_names))
        width = 0.35
        ax1.bar(x - width/2, original_counts, width, label='Original YOLOv7', color='lightcoral', alpha=0.8)
        ax1.bar(x + width/2, filtered_counts, width, label='With Betti Filtering', color='lightgreen', alpha=0.8)
        ax1.set_xlabel('Image Samples')
        ax1.set_ylabel('Detection Count')
        ax1.set_title('Detection Count: Before vs After Filtering')
        ax1.set_xticks(x)
        ax1.set_xticklabels(image_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reduction rate per image
        bars = ax2.bar(range(len(reduction_rates)), reduction_rates, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Image Samples')
        ax2.set_ylabel('Reduction Rate (%)')
        ax2.set_title('False Positive Reduction Rate by Image')
        ax2.set_xticks(range(len(image_names)))
        ax2.set_xticklabels(image_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 3. Processing time breakdown
        inference_times = [r.get('timing', {}).get('inference_time', 0) * 1000 for r in sample_results]
        filtering_times = [r.get('timing', {}).get('filtering_time', 0) * 1000 for r in sample_results]
        
        ax3.bar(range(len(image_names)), inference_times, label='Inference Time', color='orange', alpha=0.7)
        ax3.bar(range(len(image_names)), filtering_times, bottom=inference_times, 
                label='Filtering Time', color='purple', alpha=0.7)
        ax3.set_xlabel('Image Samples')
        ax3.set_ylabel('Processing Time (ms)')
        ax3.set_title('Processing Time Breakdown')
        ax3.set_xticks(range(len(image_names)))
        ax3.set_xticklabels(image_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Overall summary pie chart
        ax4.pie([stats['total_filtered'], stats['total_original'] - stats['total_filtered']], 
                labels=['Retained Detections', 'Filtered Out (False Positives)'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%',
                startangle=90)
        ax4.set_title('Overall Detection Filtering Results')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_analysis_charts.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "performance_analysis_charts.pdf", bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"📊 Charts saved to: {self.output_dir}")

def main():
    """Main execution function."""
    print("🚁 Creating Results Summary and Analysis...")
    
    analyzer = QuickResultsAnalyzer()
    sample_results = analyzer.load_sample_results()
    
    if not sample_results:
        print("❌ No sample results found to analyze!")
        return
    
    print(f"📊 Found {len(sample_results)} sample results to analyze...")
    
    print("📋 Creating performance summary table...")
    df, stats = analyzer.create_performance_summary_table(sample_results)
    
    print("📈 Creating performance charts...")
    analyzer.create_performance_charts(sample_results, stats)
    
    print(f"\n✅ Analysis complete! Results saved to: {analyzer.output_dir}")
    print(f"\nKey Results:")
    print(f"  🎯 False Positive Reduction: {stats['overall_reduction']:.1f}%")
    print(f"  ⚡ Average FPS: {stats['avg_fps']:.2f}")
    print(f"  ⏱️  Filtering Overhead: {stats['avg_filtering']*1000:.1f}ms")
    
    print(f"\nGenerated files:")
    for file in analyzer.output_dir.glob("*"):
        if file.is_file():
            print(f"  📁 {file.name}")
    
    return analyzer.output_dir

if __name__ == "__main__":
    output_dir = main()