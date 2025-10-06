import torch
import torch.profiler
import os

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("profiling/traces/baseline_trace.json")
