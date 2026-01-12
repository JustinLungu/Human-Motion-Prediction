"""Progress tracking utilities for long-running operations."""

from __future__ import annotations

import time


class ProgressTracker:
    """Track experiment progress and estimate time remaining."""
    
    def __init__(self, total_experiments: int):
        """Initialize progress tracker.
        
        Args:
            total_experiments: Total number of experiments/tasks to complete
        """
        self.total = total_experiments
        self.completed = 0
        self.start_time = time.time()
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress and print status.
        
        Args:
            increment: Number of tasks completed (default: 1)
            message: Optional message to display
        """
        self.completed += increment
        elapsed = time.time() - self.start_time
        
        if self.completed > 0:
            progress_pct = (self.completed / self.total) * 100
            avg_time_per_exp = elapsed / self.completed
            estimated_remaining = avg_time_per_exp * (self.total - self.completed)
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(estimated_remaining)
            print(f"\n[{progress_pct:5.1f}%] Completed: {self.completed}/{self.total} | "
                  f"Elapsed: {elapsed_str} | Est. remaining: {remaining_str}")
            if message:
                print(f"  {message}")
        else:
            print(f"\n[  0.0%] Starting experiments... (Total: {self.total})")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (e.g., "2m 15s", "1h 30m")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    
    def finish(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        print(f"\n[100.0%] All experiments completed in {self._format_time(total_time)}")

