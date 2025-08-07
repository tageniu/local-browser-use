#!/usr/bin/env python3
"""
WebVoyager Results Viewer
View progress and results of WebVoyager benchmark runs
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

class ResultsViewer:
    """View and analyze WebVoyager benchmark results"""
    
    def __init__(self, run_name: Optional[str] = None):
        self.results_base = Path(__file__).parent / "results"
        
        if run_name:
            self.run_dir = self.results_base / run_name
        else:
            # Find most recent run
            runs = sorted([d for d in self.results_base.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
            if runs:
                self.run_dir = runs[0]
            else:
                print("No runs found")
                return
        
        self.raw_dir = self.run_dir / "raw"
        self.aggregated_dir = self.run_dir / "aggregated"
        
    def list_runs(self):
        """List all available runs"""
        print("\nAvailable runs:")
        print("-" * 60)
        
        runs = sorted([d for d in self.results_base.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
        
        for run_dir in runs:
            # Count results
            if (run_dir / "raw").exists():
                result_count = len(list((run_dir / "raw").glob("*.json")))
            else:
                result_count = 0
                
            # Get timestamp
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
            
            print(f"  {run_dir.name:<40} {result_count:>4} tasks  {mtime:%Y-%m-%d %H:%M}")
    
    def show_progress(self):
        """Show current progress of the run"""
        
        if not self.raw_dir.exists():
            print("No results found")
            return
            
        print(f"\nProgress for run: {self.run_dir.name}")
        print("=" * 60)
        
        # Load all results
        results = []
        for result_file in self.raw_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
        
        if not results:
            print("No completed tasks yet")
            return
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.get('success'))
        failed = total - successful
        timeouts = sum(1 for r in results if r.get('timeout'))
        crashes = sum(1 for r in results if r.get('crashed'))
        no_answer = sum(1 for r in results if r.get('failure_reason') == 'no_answer_provided')
        not_grounded = sum(1 for r in results if r.get('failure_reason') == 'not_browser_grounded')
        max_steps_exceeded = sum(1 for r in results if r.get('failure_reason') == 'max_steps_exceeded')
        
        total_time = sum(r.get('time_seconds', 0) for r in results)
        avg_time = total_time / total if total > 0 else 0
        
        total_steps = sum(r.get('steps_taken', 0) for r in results)
        avg_steps = total_steps / total if total > 0 else 0
        
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"Completed tasks: {total}")
        print(f"Successful: {successful} ({success_rate:.1f}%)")
        print(f"Failed: {failed}")
        print(f"  - Timeouts: {timeouts}")
        print(f"  - Crashes: {crashes}")
        print(f"  - No answer provided: {no_answer}")
        print(f"  - Not browser grounded: {not_grounded}")
        print(f"  - Max steps exceeded: {max_steps_exceeded}")
        print(f"Average steps per task: {avg_steps:.1f}")
        print(f"Average time per task: {avg_time:.1f} seconds")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Show by category
        print("\nBy Category:")
        print("-" * 60)
        
        category_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'total_steps': 0})
        for result in results:
            web_name = result.get('web_name', result['task_id'].split('--')[0])
            category_stats[web_name]['total'] += 1
            category_stats[web_name]['total_steps'] += result.get('steps_taken', 0)
            if result.get('success'):
                category_stats[web_name]['success'] += 1
        
        print(f"{'Category':<20} {'Total':>8} {'Success':>8} {'Rate':>8} {'Avg Steps':>10}")
        print("-" * 56)
        
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_cat_steps = (stats['total_steps'] / stats['total']) if stats['total'] > 0 else 0
            print(f"{category:<20} {stats['total']:>8} {stats['success']:>8} {rate:>7.1f}% {avg_cat_steps:>10.1f}")
    
    def show_failures(self, limit: int = 20):
        """Show recent failures"""
        
        if not self.raw_dir.exists():
            print("No results found")
            return
            
        print(f"\nRecent failures in: {self.run_dir.name}")
        print("=" * 60)
        
        # Load failed results
        failures = []
        for result_file in self.raw_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)
                if not result.get('success'):
                    failures.append(result)
        
        if not failures:
            print("No failures found!")
            return
        
        # Sort by timestamp (most recent first)
        failures.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Show failures
        for i, failure in enumerate(failures[:limit], 1):
            print(f"\n{i}. {failure['task_id']}")
            print(f"   Question: {failure.get('question', 'N/A')[:100]}...")
            
            if failure.get('timeout'):
                print(f"   Status: TIMEOUT")
            elif failure.get('crashed'):
                print(f"   Status: CRASHED")
            elif failure.get('failure_reason'):
                print(f"   Status: FAILED - {failure['failure_reason']}")
            else:
                print(f"   Status: FAILED")
                
            if failure.get('error'):
                print(f"   Error: {failure['error'][:200]}")
                
            print(f"   Steps: {failure.get('steps_taken', 0)}")
            print(f"   Retries: {failure.get('retries', 0)}")
            print(f"   Time: {failure.get('time_seconds', 0):.1f}s")
    
    def show_report(self):
        """Display the aggregated report"""
        
        report_path = self.aggregated_dir / "report.md"
        
        if not report_path.exists():
            print(f"No report found. Run aggregation first.")
            return
            
        with open(report_path, 'r') as f:
            print(f.read())
    
    def export_for_experiment_results(self):
        """Export results in format compatible with experiment_results.md"""
        
        if not self.raw_dir.exists():
            print("No results found")
            return
            
        print(f"\nExporting results for: {self.run_dir.name}")
        
        # Load all results
        results = []
        for result_file in self.raw_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
        
        # Calculate overall metrics
        total = len(results)
        successful = sum(1 for r in results if r.get('success'))
        total_time = sum(r.get('time_seconds', 0) for r in results)
        total_steps = sum(r.get('steps_taken', 0) for r in results)
        
        # Generate export
        export = []
        export.append("### Baseline Performance (main.py unmodified)")
        export.append("| Metric | Value | Notes |")
        export.append("|--------|-------|-------|")
        export.append(f"| **Success Rate** | {successful}/{total} ({successful/total*100:.1f}%) | % of tasks completed successfully |")
        export.append(f"| **Avg Steps** | {total_steps/total:.1f} | Average actions per task |")
        export.append(f"| **Avg Time** | {total_time/total:.1f}s | Average seconds per task |")
        
        # Calculate error rate (tasks with retries or failures)
        error_count = sum(1 for r in results if r.get('retries', 0) > 0 or not r.get('success'))
        export.append(f"| **Error Rate** | {error_count}/{total} ({error_count/total*100:.1f}%) | % of actions that failed |")
        
        # Save export
        export_path = self.aggregated_dir / "experiment_results_export.md"
        with open(export_path, 'w') as f:
            f.write('\n'.join(export))
        
        print(f"Export saved to: {export_path}")
        print("\nContent:")
        print('\n'.join(export))


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='View WebVoyager benchmark results')
    parser.add_argument('--run', type=str, help='Specific run to view (default: most recent)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List runs
    subparsers.add_parser('list', help='List all available runs')
    
    # Show progress
    subparsers.add_parser('progress', help='Show current progress')
    
    # Show failures
    failures_parser = subparsers.add_parser('failures', help='Show recent failures')
    failures_parser.add_argument('--limit', type=int, default=20, help='Number of failures to show')
    
    # Show report
    subparsers.add_parser('report', help='Show aggregated report')
    
    # Export for experiment_results.md
    subparsers.add_parser('export', help='Export results for experiment_results.md')
    
    args = parser.parse_args()
    
    # Default to progress if no command specified
    if not args.command:
        args.command = 'progress'
    
    # Create viewer
    viewer = ResultsViewer(args.run)
    
    # Execute command
    if args.command == 'list':
        viewer.list_runs()
    elif args.command == 'progress':
        viewer.show_progress()
    elif args.command == 'failures':
        viewer.show_failures(args.limit)
    elif args.command == 'report':
        viewer.show_report()
    elif args.command == 'export':
        viewer.export_for_experiment_results()


if __name__ == "__main__":
    main()