#!/usr/bin/env python3
"""
Individual task runner for WebVoyager benchmark
Executes a single task using main.py and captures results
"""

import asyncio
import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import uuid

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

class TaskRunner:
    """Runs individual WebVoyager tasks using main.py"""
    
    def __init__(self, task_data: Dict[str, Any], results_dir: str, max_retries: int = 3, detailed_logs_dir: Optional[str] = None, run_name: Optional[str] = None):
        self.task = task_data
        self.task_id = task_data['id']
        self.question = task_data['ques']
        self.start_url = task_data['web']
        self.web_name = task_data['web_name']
        self.results_dir = Path(results_dir)
        self.max_retries = max_retries
        self.detailed_logs_dir = Path(detailed_logs_dir) if detailed_logs_dir else None
        self.run_name = run_name or "webvoyager_run"
        
        # Generate trace ID for this task
        self.trace_id = str(uuid.uuid4())
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.detailed_logs_dir:
            self.detailed_logs_dir.mkdir(parents=True, exist_ok=True)
        
    async def run_task(self) -> Dict[str, Any]:
        """Execute the task and return results"""
        
        result = {
            "task_id": self.task_id,
            "web_name": self.web_name,
            "question": self.question,
            "start_url": self.start_url,
            "success": False,
            "time_seconds": 0,
            "steps_taken": 0,
            "retries": 0,
            "error": None,
            "final_url": None,
            "answer_provided": None,
            "browser_grounded": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "crashed": False,
            "timeout": False,
            "failure_reason": None
        }
        
        for retry in range(self.max_retries):
            if retry > 0:
                print(f"[TaskRunner] Retry {retry}/{self.max_retries} for task {self.task_id}")
                result["retries"] = retry
                await asyncio.sleep(2)  # Brief pause before retry
            
            try:
                # Run the task
                task_result = await self._execute_task()
                
                # Update result with task execution data
                result.update(task_result)
                
                # If successful or non-recoverable error, stop retrying
                if result["success"] or not result["crashed"]:
                    break
                    
            except Exception as e:
                print(f"[TaskRunner] Error in task {self.task_id}: {e}")
                result["error"] = str(e)
                result["crashed"] = True
        
        # Save result to file
        result_path = self.results_dir / f"{self.task_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    async def _execute_task(self) -> Dict[str, Any]:
        """Execute main.py with the task parameters"""
        
        start_time = time.time()
        
        # Create a temporary Python script that modifies main.py behavior
        script_content = f'''
import sys
import os
sys.path.insert(0, "{Path(__file__).parent.parent}")

# Monkey-patch the main function to use our task
import main as original_main
import asyncio

async def run_webvoyager_task():
    """Run the WebVoyager task"""
    
    task_description = """{self.question}"""
    start_url = """{self.start_url}"""
    
    # Enable detailed logging if directory is provided
    detailed_logs_dir = {repr(str(self.detailed_logs_dir)) if self.detailed_logs_dir else 'None'}
    enable_logging = detailed_logs_dir is not None
    
    # Use Langfuse tracing instead of file logging when possible
    agent = original_main.BrowserAgent(
        task=task_description,
        enable_detailed_logging=enable_logging,
        log_base_dir=detailed_logs_dir,
        trace_id="{self.trace_id}",
        session_id="{self.run_name}",
        user_id="{self.task_id}"
    )
    result = await agent.run(start_url=start_url)
    
    # Print result as JSON for parsing
    import json
    print("RESULT_START")
    print(json.dumps(result))
    print("RESULT_END")
    
    return result

if __name__ == "__main__":
    asyncio.run(run_webvoyager_task())
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            # Run the script with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ}
            )
            
            # Wait for completion with timeout (20 minutes)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=20 * 60  # 20 minutes
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "timeout": True,
                    "time_seconds": time.time() - start_time,
                    "error": "Task timed out after 20 minutes",
                    "failure_reason": "timeout_exceeded"
                }
            
            # Parse output
            output = stdout.decode('utf-8')
            
            # Extract result JSON
            if "RESULT_START" in output and "RESULT_END" in output:
                result_json = output.split("RESULT_START")[1].split("RESULT_END")[0].strip()
                task_result = json.loads(result_json)
                
                # Check WebVoyager success criteria
                extracted_data = task_result.get("extracted_data", None)
                steps = task_result.get("steps", 0)
                
                # Determine if answer is browser-grounded (has extracted data and visited pages)
                browser_grounded = (
                    extracted_data is not None and 
                    task_result.get("final_url", "") != self.start_url
                )
                
                # Add trace ID to result for reference
                task_result["langfuse_trace_id"] = self.trace_id
                
                # Check for various failure modes
                failure_reason = None
                success = task_result.get("success", False)
                
                if not success:
                    if steps >= 30:  # MAX_STEPS in main.py
                        failure_reason = "max_steps_exceeded"
                    elif not extracted_data:
                        failure_reason = "no_answer_provided"
                    elif not browser_grounded:
                        failure_reason = "not_browser_grounded"
                    else:
                        failure_reason = "task_incomplete"
                
                return {
                    "success": success,
                    "time_seconds": time.time() - start_time,
                    "steps_taken": steps,
                    "final_url": task_result.get("final_url", None),
                    "extracted_data": extracted_data,
                    "answer_provided": extracted_data is not None,
                    "browser_grounded": browser_grounded,
                    "history": task_result.get("history", []),
                    "crashed": False,
                    "failure_reason": failure_reason,
                    "langfuse_trace_id": self.trace_id
                }
            else:
                # Could not parse output, check for errors
                error_output = stderr.decode('utf-8') if stderr else ""
                return {
                    "success": False,
                    "time_seconds": time.time() - start_time,
                    "error": f"Could not parse output. Stderr: {error_output[:500]}",
                    "crashed": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "time_seconds": time.time() - start_time,
                "error": str(e),
                "crashed": True
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_script):
                os.unlink(temp_script)


async def main():
    """Main entry point for running a single task"""
    
    parser = argparse.ArgumentParser(description='Run a single WebVoyager task')
    parser.add_argument('--task-json', required=True, help='JSON string of the task')
    parser.add_argument('--results-dir', default='results/raw', help='Directory to save results')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries on crash')
    parser.add_argument('--detailed-logs-dir', help='Directory for detailed action logs')
    parser.add_argument('--run-name', help='Name for this run (used for Langfuse session ID)')
    
    args = parser.parse_args()
    
    # Parse task JSON
    task_data = json.loads(args.task_json)
    
    # Create runner
    runner = TaskRunner(task_data, args.results_dir, args.max_retries, args.detailed_logs_dir, args.run_name)
    
    # Run task
    print(f"[TaskRunner] Starting task {task_data['id']}: {task_data['ques'][:80]}...")
    result = await runner.run_task()
    
    # Print summary
    if result["success"]:
        print(f"[TaskRunner] ✓ Task {task_data['id']} completed successfully in {result['time_seconds']:.1f}s")
    else:
        print(f"[TaskRunner] ✗ Task {task_data['id']} failed after {result['retries']} retries")
        if result.get("error"):
            print(f"[TaskRunner]   Error: {result['error'][:200]}")
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())