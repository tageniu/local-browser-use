#!/usr/bin/env python3
"""
WebVoyager Benchmark Runner - Main Orchestrator
Manages parallel execution of all 590 WebVoyager tasks using main.py
"""

import asyncio
import json
import os
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import pytz
from typing import Dict, List, Any, Optional
from collections import defaultdict
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re

class WebVoyagerRunner:
    """Orchestrates the execution of WebVoyager benchmark tasks"""
    
    def __init__(self, workers: int = 4, run_name: Optional[str] = None, 
                 resume: bool = False, categories: Optional[List[str]] = None,
                 enable_detailed_logging: bool = True):
        self.workers = workers
        self.data_path = Path(__file__).parent / "data" / "patchedTasks.jsonl"
        
        # Extract model information from main.py
        model_name = self._get_model_from_main()
        
        # Create run directory with unique name
        if run_name:
            self.run_name = run_name
        else:
            # Get Pacific Time (Los Angeles)
            pacific = pytz.timezone('America/Los_Angeles')
            now_pacific = datetime.now(pacific)
            
            # Format: YYYYMMDD_HHMM_PT_{model_name}
            timestamp = now_pacific.strftime('%Y%m%d_%H%M_PT')
            
            # Clean model name for filename (remove special chars)
            clean_model = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
            
            self.run_name = f"{timestamp}_{clean_model}"
        
        self.results_dir = Path(__file__).parent / "results" / self.run_name
        self.raw_results_dir = self.results_dir / "raw"
        self.aggregated_dir = self.results_dir / "aggregated"
        self.detailed_logs_dir = self.results_dir / "detailed_logs"  # New directory for detailed logs
        
        # Create directories
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_logs_dir.mkdir(parents=True, exist_ok=True)  # Create detailed logs directory
        
        # Set attributes before saving metadata
        self.resume = resume
        self.categories = categories
        self.enable_detailed_logging = enable_detailed_logging
        
        # Save metadata about the run (after setting attributes)
        self._save_run_metadata(model_name)
        
        # Log run info
        print(f"[Runner] Run Name: {self.run_name}")
        print(f"[Runner] Langfuse traces will be under session: {self.run_name}")
        
        # Track progress
        self.total_tasks = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
    
    def _get_model_from_main(self) -> str:
        """Extract the model name being used from main.py"""
        main_py_path = Path(__file__).parent.parent / "main.py"
        
        try:
            with open(main_py_path, 'r') as f:
                content = f.read()
                
            # First check LLM_PROVIDER
            provider_match = re.search(r'LLM_PROVIDER\s*=\s*os\.getenv\("LLM_PROVIDER",\s*"([^"]+)"\)', content)
            provider = provider_match.group(1) if provider_match else "groq"
            
            # Then get MODEL_NAME
            model_match = re.search(r'MODEL_NAME\s*=\s*os\.getenv\("MODEL_NAME",\s*"([^"]+)"\)', content)
            model = model_match.group(1) if model_match else "unknown"
            
            # Check if there's an env override
            env_provider = os.getenv("LLM_PROVIDER")
            env_model = os.getenv("MODEL_NAME")
            
            if env_provider:
                provider = env_provider
            if env_model:
                model = env_model
            
            return f"{provider}_{model}"
            
        except Exception as e:
            print(f"[Runner] Warning: Could not extract model from main.py: {e}")
            return "unknown_model"
    
    def _save_run_metadata(self, model_name: str):
        """Save metadata about this run"""
        metadata = {
            "run_name": self.run_name,
            "model": model_name,
            "workers": self.workers,
            "started_at": datetime.now().isoformat(),
            "started_at_pacific": datetime.now(pytz.timezone('America/Los_Angeles')).isoformat(),
            "categories": self.categories,
            "resume": self.resume,
            "detailed_logging_enabled": self.enable_detailed_logging,
            "langfuse_session": self.run_name
        }
        
        metadata_path = self.results_dir / "run_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load all tasks from JSONL file"""
        tasks = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                task = json.loads(line.strip())
                
                # Filter by category if specified
                if self.categories and task['web_name'] not in self.categories:
                    continue
                    
                # Skip if resuming and task already completed
                if self.resume:
                    result_file = self.raw_results_dir / f"{task['id']}.json"
                    if result_file.exists():
                        # Check if task was successful or timed out
                        with open(result_file, 'r') as rf:
                            result = json.load(rf)
                            if result.get('success') or result.get('timeout'):
                                print(f"[Runner] Skipping completed task: {task['id']}")
                                continue
                
                tasks.append(task)
        
        self.total_tasks = len(tasks)
        return tasks
    
    def run_task_subprocess(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task in a subprocess"""
        
        task_json = json.dumps(task)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "task_runner.py"),
            "--task-json", task_json,
            "--results-dir", str(self.raw_results_dir),
            "--max-retries", "3",
            "--run-name", self.run_name  # Pass run name for Langfuse session ID
        ]
        
        # Add detailed logging arguments if enabled (fallback to file logging)
        if self.enable_detailed_logging:
            cmd.extend(["--detailed-logs-dir", str(self.detailed_logs_dir)])
        
        try:
            # Run task with overall timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=25 * 60  # 25 minutes total timeout
            )
            
            # Load the result file
            result_file = self.raw_results_dir / f"{task['id']}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "task_id": task['id'],
                    "success": False,
                    "error": "Result file not created",
                    "crashed": True
                }
                
        except subprocess.TimeoutExpired:
            return {
                "task_id": task['id'],
                "success": False,
                "timeout": True,
                "error": "Overall task timeout (25 minutes)"
            }
        except Exception as e:
            return {
                "task_id": task['id'],
                "success": False,
                "error": str(e),
                "crashed": True
            }
    
    def print_progress(self):
        """Print current progress"""
        if self.completed_tasks == 0:
            return
            
        elapsed = time.time() - self.start_time
        rate = self.completed_tasks / elapsed
        eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
        
        success_rate = (self.successful_tasks / self.completed_tasks * 100) if self.completed_tasks > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Progress: {self.completed_tasks}/{self.total_tasks} tasks completed")
        print(f"Success rate: {self.successful_tasks}/{self.completed_tasks} ({success_rate:.1f}%)")
        print(f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        print(f"{'='*60}\n")
    
    def run_parallel(self, tasks: List[Dict[str, Any]]):
        """Run tasks in parallel using multiple workers"""
        
        self.start_time = time.time()
        
        print(f"[Runner] Starting {len(tasks)} tasks with {self.workers} workers")
        print(f"[Runner] Results will be saved to: {self.results_dir}")
        
        # Track timing for ETA calculation
        task_times = []
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_task_subprocess, task): task 
                for task in tasks
            }
            
            # Create progress bar with rich information
            with tqdm(total=len(tasks), desc="Running tasks", unit="task", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                     ncols=100) as pbar:
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    task_start = time.time()
                    
                    try:
                        result = future.result()
                        self.completed_tasks += 1
                        
                        # Track task completion time
                        task_time = time.time() - self.start_time
                        task_times.append(task_time)
                        
                        if result.get('success'):
                            self.successful_tasks += 1
                            status = "✓"
                            status_color = "green"
                        else:
                            self.failed_tasks += 1
                            status = "✗"
                            status_color = "red"
                            if result.get('timeout'):
                                status += " (timeout)"
                            elif result.get('crashed'):
                                status += " (crashed)"
                        
                        time_str = f"{result.get('time_seconds', 0):.1f}s" if result.get('time_seconds') else "N/A"
                        
                        # Calculate success rate
                        success_rate = (self.successful_tasks / self.completed_tasks * 100) if self.completed_tasks > 0 else 0
                        
                        # Update progress bar with detailed information
                        pbar.set_postfix({
                            'Success': f'{self.successful_tasks}/{self.completed_tasks} ({success_rate:.1f}%)',
                            'Last': f'{task["id"][:20]} {status}'
                        })
                        pbar.update(1)
                        
                        # Also print detailed info for each task (will appear above progress bar)
                        tqdm.write(f"[{self.completed_tasks}/{self.total_tasks}] {status} {task['id']} - {time_str}")
                        
                    except Exception as e:
                        tqdm.write(f"[Runner] Error processing task {task['id']}: {e}")
                        self.completed_tasks += 1
                        self.failed_tasks += 1
                        pbar.update(1)
                
                # Final summary after progress bar completes
                elapsed = time.time() - self.start_time
                success_rate = (self.successful_tasks / self.completed_tasks * 100) if self.completed_tasks > 0 else 0
                
                print(f"\n{'='*60}")
                print(f"Completed: {self.completed_tasks}/{self.total_tasks} tasks")
                print(f"Success rate: {self.successful_tasks}/{self.completed_tasks} ({success_rate:.1f}%)")
                print(f"Total elapsed time: {elapsed:.1f}s")
                print(f"Average time per task: {elapsed/self.completed_tasks:.1f}s")
                print(f"Langfuse Session: {self.run_name}")
                print(f"View traces at: https://us.cloud.langfuse.com")
                print(f"{'='*60}\n")
    
    def aggregate_results(self):
        """Aggregate results and generate report"""
        
        print("\n[Runner] Aggregating results...")
        
        # Load all result files
        results = []
        for result_file in self.raw_results_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
        
        # Group by category
        category_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'failed': 0,
            'timeout': 0,
            'crashed': 0,
            'no_answer': 0,
            'not_grounded': 0,
            'max_steps': 0,
            'total_time': 0,
            'total_steps': 0,
            'total_retries': 0,
            'avg_steps': 0,
            'tasks': []
        })
        
        for result in results:
            web_name = result.get('web_name', result['task_id'].split('--')[0])
            stats = category_stats[web_name]
            
            stats['total'] += 1
            stats['tasks'].append(result['task_id'])
            
            if result.get('success'):
                stats['success'] += 1
            else:
                stats['failed'] += 1
                
            if result.get('timeout'):
                stats['timeout'] += 1
            if result.get('crashed'):
                stats['crashed'] += 1
            
            # Track failure reasons
            failure_reason = result.get('failure_reason')
            if failure_reason == 'no_answer_provided':
                stats['no_answer'] += 1
            elif failure_reason == 'not_browser_grounded':
                stats['not_grounded'] += 1
            elif failure_reason == 'max_steps_exceeded':
                stats['max_steps'] += 1
                
            stats['total_time'] += result.get('time_seconds', 0)
            stats['total_steps'] += result.get('steps_taken', 0)
            stats['total_retries'] += result.get('retries', 0)
        
        # Calculate overall statistics
        total_tasks = sum(s['total'] for s in category_stats.values())
        total_success = sum(s['success'] for s in category_stats.values())
        total_time = sum(s['total_time'] for s in category_stats.values())
        total_steps = sum(s['total_steps'] for s in category_stats.values())
        
        # Calculate averages for each category
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                stats['avg_steps'] = stats['total_steps'] / stats['total']
        
        overall_success_rate = (total_success / total_tasks * 100) if total_tasks > 0 else 0
        overall_avg_steps = (total_steps / total_tasks) if total_tasks > 0 else 0
        
        # Generate report
        report = []
        report.append(f"# WebVoyager Benchmark Results - {self.run_name}")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Overall Statistics")
        report.append(f"- **Total Tasks**: {total_tasks}")
        report.append(f"- **Successful**: {total_success} ({overall_success_rate:.1f}%)")
        report.append(f"- **Failed**: {total_tasks - total_success}")
        report.append(f"- **Average Steps**: {overall_avg_steps:.1f}")
        report.append(f"- **Total Time**: {total_time:.1f} seconds")
        report.append(f"- **Average Time per Task**: {total_time/total_tasks:.1f} seconds")
        
        report.append(f"\n## Results by Category\n")
        report.append("| Category | Total | Success | Success Rate | Avg Steps | Avg Time | Timeouts | No Answer | Not Grounded |")
        report.append("|----------|-------|---------|--------------|-----------|----------|----------|-----------|--------------|")
        
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
            
            report.append(f"| {category} | {stats['total']} | {stats['success']} | "
                         f"{success_rate:.1f}% | {stats['avg_steps']:.1f} | {avg_time:.1f}s | "
                         f"{stats['timeout']} | {stats['no_answer']} | {stats['not_grounded']} |")
        
        # Add detailed task results
        report.append(f"\n## Individual Task Results\n")
        
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            report.append(f"\n### {category}")
            
            # Load and sort task results
            category_results = []
            for task_id in stats['tasks']:
                result_file = self.raw_results_dir / f"{task_id}.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        category_results.append(json.load(f))
            
            category_results.sort(key=lambda x: x['task_id'])
            
            for result in category_results:
                status = "✓" if result.get('success') else "✗"
                time_str = f"{result.get('time_seconds', 0):.1f}s" if result.get('time_seconds') else "N/A"
                
                steps_str = f"{result.get('steps_taken', 0)} steps"
                
                details = f"{status} **{result['task_id']}** - {steps_str}, {time_str}"
                if result.get('timeout'):
                    details += " (timeout)"
                elif result.get('crashed'):
                    details += " (crashed)"
                elif result.get('failure_reason'):
                    details += f" ({result['failure_reason']})"
                elif result.get('retries', 0) > 0:
                    details += f" ({result['retries']} retries)"
                    
                report.append(f"- {details}")
        
        # Save report
        report_path = self.aggregated_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"[Runner] Report saved to: {report_path}")
        
        # Save aggregated statistics as JSON
        stats_path = self.aggregated_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'run_name': self.run_name,
                'timestamp': datetime.now().isoformat(),
                'overall': {
                    'total_tasks': total_tasks,
                    'successful': total_success,
                    'failed': total_tasks - total_success,
                    'success_rate': overall_success_rate,
                    'average_steps': overall_avg_steps,
                    'total_time_seconds': total_time
                },
                'by_category': dict(category_stats)
            }, f, indent=2)
        
        print(f"[Runner] Statistics saved to: {stats_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - {self.run_name}")
        print(f"{'='*60}")
        print(f"Total Tasks: {total_tasks}")
        print(f"Successful: {total_success} ({overall_success_rate:.1f}%)")
        print(f"Failed: {total_tasks - total_success}")
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"{'='*60}")
    
    def run(self):
        """Main execution method"""
        
        # Load tasks
        tasks = self.load_tasks()
        
        if not tasks:
            print("[Runner] No tasks to run")
            return
        
        print(f"[Runner] Loaded {len(tasks)} tasks to run")
        
        # Run tasks in parallel
        self.run_parallel(tasks)
        
        # Aggregate results
        self.aggregate_results()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run WebVoyager benchmark tests')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--run-name', type=str, 
                       help='Name for this run (default: timestamp)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories to run (e.g., Allrecipes Amazon)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of tasks to run (for testing)')
    parser.add_argument('--disable-detailed-logging', action='store_true',
                       help='Disable detailed action logging (enabled by default)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = WebVoyagerRunner(
        workers=args.workers,
        run_name=args.run_name,
        resume=args.resume,
        categories=args.categories,
        enable_detailed_logging=not args.disable_detailed_logging
    )
    
    # Run benchmark
    runner.run()


if __name__ == "__main__":
    main()