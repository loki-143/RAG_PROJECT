# CPU Lag Fix Guide

## Problem

Parallel embedding was causing **severe CPU lag** and system unresponsiveness, especially on i3/i5 CPUs.

## Root Cause

The parallel embedding implementation was too aggressive:
- **4 workers** running simultaneously
- Each worker running CPU-intensive embedding operations
- **100% CPU usage** on all cores
- System became unresponsive

## Solution Applied

### 1. Conservative Worker Count

**Before:**
```python
max_workers = min(4, cpu_count)  # Always use 4 workers
```

**After:**
```python
if cpu_count <= 4:
    max_workers = 2  # i3/i5: Only 2 workers
elif cpu_count <= 8:
    max_workers = 3  # i7: 3 workers
else:
    max_workers = 4  # i9+: 4 workers
```

### 2. Higher Threshold for Parallel Processing

**Before:**
```python
if len(chunks) > 1000:  # Use parallel for >1000 chunks
    parallel_embed()
```

**After:**
```python
if len(chunks) > 5000:  # Use parallel only for >5000 chunks
    parallel_embed()
else:
    sequential_embed()  # Better system responsiveness
```

### 3. Disabled by Default

**Before:**
```bash
USE_PARALLEL_EMBEDDINGS=true  # Enabled by default
```

**After:**
```bash
USE_PARALLEL_EMBEDDINGS=false  # Disabled by default for safety
```

## Configuration

### For Your i3 12th Gen System

**Recommended settings in `.env`:**

```bash
# DISABLE parallel embeddings (better system responsiveness)
USE_PARALLEL_EMBEDDINGS=false

# If you want to try parallel, use high threshold
PARALLEL_EMBEDDING_THRESHOLD=10000

# Keep batch size moderate
EMBEDDING_BATCH_SIZE=500
```

### For Different Systems

**Low-end (i3, i5, Ryzen 3, Ryzen 5):**
```bash
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000
EMBEDDING_BATCH_SIZE=500
```

**Mid-range (i7, Ryzen 7):**
```bash
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=5000
EMBEDDING_BATCH_SIZE=750
```

**High-end (i9, Ryzen 9, Threadripper):**
```bash
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
EMBEDDING_BATCH_SIZE=1000
```

## Performance Comparison

### Your 81,166 Chunk Repository

**Sequential (Recommended for i3):**
```
Time: ~45 minutes
CPU Usage: 50-70% (one core)
System: Responsive ✅
Can use computer normally ✅
```

**Parallel (2 workers):**
```
Time: ~25 minutes
CPU Usage: 90-100% (all cores)
System: Sluggish ⚠️
Hard to use computer ❌
```

**Parallel (4 workers):**
```
Time: ~20 minutes
CPU Usage: 100% (all cores)
System: Frozen/Unresponsive ❌
Cannot use computer ❌
```

## Why Sequential is Better for You

### Advantages
- ✅ System stays responsive
- ✅ Can use computer during indexing
- ✅ No overheating issues
- ✅ More stable
- ✅ Lower power consumption

### Disadvantages
- ⏱️ Takes longer (~45 min vs ~20 min)

### The Trade-off

**For your use case:**
- You're indexing occasionally, not constantly
- System responsiveness > speed
- 45 minutes is acceptable for a one-time operation
- You can work on other things while it runs

**Sequential is the right choice!**

## How to Apply the Fix

### Option 1: Use Sequential (Recommended)

Already set in `.env`:
```bash
USE_PARALLEL_EMBEDDINGS=false
```

Just restart your server:
```bash
# Stop server (Ctrl+C)
python backend/fastapi_app.py
```

### Option 2: Try Conservative Parallel

If you want to try parallel with reduced lag:

```bash
# In .env
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=10000  # Very high threshold
```

This will:
- Use sequential for repos <10K chunks
- Use only 2 workers for larger repos
- Still cause some lag, but less severe

## Monitoring

### Check What Mode is Being Used

Look for these logs:

**Sequential mode:**
```
INFO - Using sequential embedding (<=5000 chunks) for better system responsiveness
INFO - Embedding batch 1/163: 500 chunks
```

**Parallel mode:**
```
INFO - Using parallel embedding (>5000 chunks). System may be slower during this process.
INFO - Using parallel embedding with 2 workers (CPU cores: 4, batch_size=500)
```

### Monitor CPU Usage

**Windows:**
- Open Task Manager (Ctrl+Shift+Esc)
- Watch CPU usage during embedding
- Should be 50-70% for sequential
- Will be 90-100% for parallel

**If CPU is 100% and system is laggy:**
- Stop the indexing (Ctrl+C)
- Set `USE_PARALLEL_EMBEDDINGS=false`
- Restart and try again

## Troubleshooting

### Issue: Still Laggy with Sequential

**Possible causes:**
1. Other programs using CPU
2. Antivirus scanning
3. Background Windows updates

**Solutions:**
```bash
# Reduce batch size (uses less memory, less CPU spikes)
EMBEDDING_BATCH_SIZE=250

# Close other programs
# Disable antivirus temporarily
# Pause Windows updates
```

### Issue: Want Faster Indexing Without Lag

**Options:**

1. **Run overnight:**
   ```bash
   # Start indexing before bed
   # Let it run overnight
   # Wake up to completed index
   ```

2. **Use GPU (if available):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # 10-50x faster, no CPU lag
   ```

3. **Upgrade CPU:**
   - i7 or Ryzen 7: Better parallel performance
   - More cores = less lag with parallel

### Issue: Parallel Still Enabled

**Check your .env file:**
```bash
# Should be:
USE_PARALLEL_EMBEDDINGS=false

# NOT:
USE_PARALLEL_EMBEDDINGS=true
```

**Restart server after changing:**
```bash
# Stop (Ctrl+C)
python backend/fastapi_app.py
```

## Best Practices

### For Development (Your Use Case)

```bash
# Prioritize system responsiveness
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000
EMBEDDING_BATCH_SIZE=500
```

**Why:**
- You need to use your computer while indexing
- Occasional indexing, not production workload
- 45 minutes is acceptable

### For Production Server

```bash
# Prioritize speed (dedicated server)
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
EMBEDDING_BATCH_SIZE=1000
```

**Why:**
- Dedicated server, no other work
- Speed is critical
- System lag doesn't matter

### For Laptop

```bash
# Avoid overheating and battery drain
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=20000
EMBEDDING_BATCH_SIZE=500
```

**Why:**
- Laptops overheat easily
- Battery drains fast with 100% CPU
- Thermal throttling makes parallel slower anyway

## Summary

### What Changed

✅ **Conservative worker count** (2 workers for i3/i5)  
✅ **Higher threshold** (5000 → only for large repos)  
✅ **Disabled by default** (opt-in for parallel)  
✅ **Better logging** (shows which mode is used)  

### Recommended for You

```bash
USE_PARALLEL_EMBEDDINGS=false
```

This gives you:
- ✅ Responsive system
- ✅ Can work during indexing
- ✅ Stable performance
- ⏱️ 45 minutes for 81K chunks (acceptable)

### When to Use Parallel

Only if:
- You have i7/i9 or Ryzen 7/9
- You're okay with system lag
- You need maximum speed
- You won't use computer during indexing

**For your i3 system: Sequential is the right choice!** 🎯
