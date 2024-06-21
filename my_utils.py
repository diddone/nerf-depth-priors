import time
import gc
import torch

class MemLogger:
  def __init__(self, is_active=True):
        self.is_active = is_active

  def log_mem(self, step, msg: str):
      if self.is_active:
        if step % 1024 == 1:
            import gc
            gc.collect()
            torch.cuda.synchronize()
            print(f"Step {step} ", msg)
            print(int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "MB", flush=True)
            print(int(torch.cuda.memory_reserved() / (1024 * 1024)), flush=True)
            print("--------------------------------------------------------", flush=True)

            torch.cuda.reset_max_memory_allocated()

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.total_elapsed_time = 0

    def __enter__(self):
        self.stop_time = time.time()
        self.total_elapsed_time += self.stop_time - self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.start_time = time.time()

    def elapsed(self):
        current_time = time.time()
        print("ELAPSED TRAINING TIME", self.total_elapsed_time + (current_time - self.start_time))

