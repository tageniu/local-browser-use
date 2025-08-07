# WebVoyager Benchmark Replication System

This directory contains the infrastructure for replicating the WebVoyager benchmark experiments using the `main.py` browser automation agent.

## Overview

The system runs all 590 WebVoyager tasks in parallel, tracking success rates, execution time, and retry statistics. Results are stored individually and aggregated for analysis.

## Files

- `webvoyager_runner.py` - Main orchestrator that manages parallel execution
- `task_runner.py` - Individual task executor that calls main.py
- `view_results.py` - Results viewer and progress monitor
- `data/patchedTasks.jsonl` - WebVoyager task dataset (590 tasks)

## Quick Start

### Run Full Benchmark (All 590 tasks)
```bash
python webvoyager_runner.py --workers 4
```

### Run Specific Categories
```bash
# Run only Allrecipes and Amazon tasks
python webvoyager_runner.py --categories Allrecipes Amazon --workers 2
```

### Test Run (Limited tasks)
```bash
# Run first 10 tasks for testing
python webvoyager_runner.py --limit 10 --workers 2 --run-name test_run
```

### Resume Interrupted Run
```bash
python webvoyager_runner.py --resume --run-name baseline_run_20250806_120000
```

## Monitoring Progress

### View Current Progress
```bash
python view_results.py progress
```

### List All Runs
```bash
python view_results.py list
```

### Show Recent Failures
```bash
python view_results.py failures --limit 10
```

### View Full Report
```bash
python view_results.py report
```

### Export for experiment_results.md
```bash
python view_results.py export
```

## Results Structure

```
results/
└── baseline_run_[timestamp]/
    ├── raw/                    # Individual task results
    │   ├── Allrecipes--0.json
    │   ├── Allrecipes--1.json
    │   └── ...
    └── aggregated/            # Summary statistics
        ├── report.md          # Human-readable report
        └── statistics.json    # Machine-readable statistics
```

## Task Result Format

Each task result contains:
```json
{
  "task_id": "Allrecipes--0",
  "web_name": "Allrecipes",
  "question": "Provide a recipe for vegetarian lasagna...",
  "start_url": "https://www.allrecipes.com/",
  "success": true,
  "time_seconds": 125.3,
  "steps_taken": 15,
  "retries": 0,
  "error": null,
  "final_url": "...",
  "timestamp": "2025-08-06 12:00:00"
}
```

## Metrics Tracked

1. **Success Rate**: Percentage of tasks completed successfully
2. **Execution Time**: Time taken per task and total
3. **Steps**: Number of browser actions per task
4. **Retries**: Number of retry attempts after crashes
5. **Failures**: Breakdown by timeout vs crash vs logic error

## Parallelization

- Default: 4 workers
- Each worker runs tasks in separate processes
- Automatic retry on crashes (up to 3 attempts)
- 20-minute timeout per task

## Categories in Dataset

The WebVoyager dataset contains tasks from these websites:
- Allrecipes (cooking recipes)
- Amazon (e-commerce)
- Apple (product information)
- ArXiv (academic papers)
- BBC News (news articles)
- Booking (hotel reservations)
- ESPN (sports information)
- GitHub (code repositories)
- Google Flights (flight search)
- Google Maps (location search)
- Google Search (web search)
- Coursera (online courses)
- Wolfram Alpha (computational queries)
- And more...

## Troubleshooting

### If tasks are failing consistently:
1. Check that `main.py` is working: `python ../main.py`
2. Verify Ollama/Groq is configured correctly
3. Check browser installation (patchright/chromium)
4. Review individual task logs in `results/[run]/raw/`

### Memory issues with many workers:
- Reduce workers: `--workers 2`
- Each worker spawns a browser instance

### To debug a specific task:
```bash
# Run single task with verbose output
python task_runner.py --task-json '{"id":"Allrecipes--0","ques":"...","web":"...","web_name":"Allrecipes"}'
```

## Expected Performance

Based on the WebVoyager paper:
- Human performance: ~90% success rate
- GPT-4V (paper baseline): ~50% success rate
- Expected runtime: 10-20 hours for full dataset with 4 workers

## Notes

- Tasks are run using the unmodified `main.py` agent
- Each task starts fresh (no memory between tasks)
- Results are saved immediately after each task completes
- The system is resilient to crashes and can be resumed