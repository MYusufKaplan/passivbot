# Stagnation Counter - Implementation Summary

## Overview

Added a **persistent stagnation counter** that tracks consecutive generations without improvement and displays in the generation panel. The counter persists through checkpoint saves and resumes.

## Features

### 1. Stagnation Counter Tracking

- **Increments**: Every check interval (default: 5 gens) when global best doesn't improve
- **Resets**: When global best improves or after diversity injection
- **Persists**: Saved in checkpoints and restored on resume
- **Displays**: Shows in every generation panel

### 2. Improvement Detection

Uses a threshold-based comparison:
```python
improvement_threshold = 1e-8  # Minimum improvement to count as progress

if global_best_fitness < prev_best - improvement_threshold:
    # Improvement detected - reset counter
    stagnation_counter = 0
else:
    # No improvement - increment counter
    stagnation_counter += check_interval
```

### 3. Display in Generation Panel

**When Improving:**
```
🌟 Gen 150
🧬 Gen Best: 1.234567e-03
🏆 Global Best: 9.876543e-04
...
✅ Improving | Injections: 2
```

**When Stagnant:**
```
🌟 Gen 175
🧬 Gen Best: 9.876543e-04
🏆 Global Best: 9.876543e-04
...
⚠️ Stagnation: 25 gens | Injections: 2
```

### 4. Checkpoint Persistence

**Saved in checkpoint:**
```python
{
    "population": population,
    "logbook": logbook,
    "generation": gen,
    "stagnation_detector": {
        'stagnation_counter': 25,
        'injection_count': 2,
        'last_improvement_gen': 150,
        'prev_global_best': 9.876543e-04,
        ...
    }
}
```

**Restored on resume:**
```
📦 Checkpoint found. Loading...
✅ Resuming from generation 176
🔄 Restored stagnation detector (injections: 2)
📊 Stagnation tracking resumed: 25 gens stagnant, 2 injections
```

## Implementation Details

### Stagnation Detector Structure

```python
stagnation_detector = {
    'hypervolume_history': [],
    'diversity_history': [],
    'fitness_std_history': [],
    'non_dom_ratio_history': [],
    'initial_diversity': None,
    'last_injection_gen': 0,
    'injection_count': 0,
    'stagnation_counter': 0,        # NEW: Consecutive stagnant gens
    'last_improvement_gen': 0,      # NEW: Last gen with improvement
    'prev_global_best': float('inf') # NEW: Previous best for comparison
}
```

### Counter Update Logic

```python
# Every check_interval generations (default: 5)
if gen % check_interval == 0 and gen > min_generation:
    # Check for improvement
    prev_best = stagnation_detector.get('prev_global_best', float('inf'))
    
    if global_best_fitness < prev_best - improvement_threshold:
        # Improvement - reset counter
        stagnation_detector['stagnation_counter'] = 0
        stagnation_detector['last_improvement_gen'] = gen
    else:
        # No improvement - increment
        stagnation_detector['stagnation_counter'] += check_interval
    
    # Store for next comparison
    stagnation_detector['prev_global_best'] = global_best_fitness
```

### Display Logic

```python
if stag_cfg['enabled']:
    stag_counter = stagnation_detector.get('stagnation_counter', 0)
    injection_count = stagnation_detector.get('injection_count', 0)
    
    if stag_counter > 0:
        status_msg += f"\n⚠️ Stagnation: {stag_counter} gens | Injections: {injection_count}"
    else:
        status_msg += f"\n✅ Improving | Injections: {injection_count}"
```

## Use Cases

### 1. Monitoring Optimization Progress

Quickly see if optimization is making progress:
- **Counter = 0**: Actively improving
- **Counter < 20**: Recent improvement
- **Counter > 50**: Long stagnation period

### 2. Tuning Injection Timing

Adjust `min_generations_between_injections` based on counter:
- If counter reaches 100+ before injection, increase injection frequency
- If injections happen too early, increase the threshold

### 3. Checkpoint Resume Verification

After resuming from checkpoint:
```
📊 Stagnation tracking resumed: 45 gens stagnant, 3 injections
```
Confirms the optimization state was properly restored.

### 4. Debugging Stagnation Issues

If counter keeps growing without triggering injection:
- Check `stagnation_score_threshold` (may be too high)
- Verify individual criteria are being met
- Review `min_generations_between_injections` setting

## Configuration

No new configuration needed! The counter uses existing settings:

```json
{
  "stagnation_detection": {
    "enabled": true,
    "check_interval": 5,  // Counter increments by this amount
    "min_generation": 10  // Counter starts tracking after this
  }
}
```

## Benefits

1. **Visibility**: Always know if optimization is progressing
2. **Persistence**: Counter survives restarts
3. **Debugging**: Helps identify stagnation patterns
4. **Tuning**: Informs parameter adjustments
5. **Monitoring**: Easy to spot long stagnation periods

## Technical Notes

### Improvement Threshold

The threshold `1e-8` is chosen to:
- Ignore floating-point noise
- Detect meaningful improvements
- Avoid false positives from tiny changes

Adjust if needed:
```python
improvement_threshold = 1e-8  # Very sensitive
improvement_threshold = 1e-6  # Moderate
improvement_threshold = 1e-4  # Only significant improvements
```

### Counter Increment

Counter increments by `check_interval` (not by 1) because:
- Checks happen every N generations
- Counter represents actual generations stagnant
- Makes the number more meaningful

Example:
- Check every 5 gens
- After 4 checks with no improvement
- Counter = 20 (not 4)

### Reset Behavior

Counter resets in two scenarios:

1. **Improvement detected**: Natural progress
2. **Diversity injection**: Fresh start after intervention

This ensures the counter accurately reflects current stagnation state.

## Future Enhancements

Potential improvements:

1. **Adaptive thresholds**: Adjust improvement threshold based on fitness scale
2. **Multiple counters**: Track different types of stagnation separately
3. **Historical tracking**: Store counter history for analysis
4. **Alerts**: Warn when counter exceeds certain thresholds
5. **Visualization**: Plot counter over time

## Troubleshooting

### Counter Not Resetting

If counter keeps growing despite improvements:
- Check `improvement_threshold` (may be too large)
- Verify `global_best_fitness` is actually improving
- Check for numerical precision issues

### Counter Always Zero

If counter never increments:
- Verify `check_interval` is being reached
- Check if `min_generation` is too high
- Ensure stagnation detection is enabled

### Counter Not Persisting

If counter resets after checkpoint resume:
- Verify checkpoint file contains `stagnation_detector`
- Check for pickle errors in logs
- Ensure checkpoint is being saved at intervals

## Example Output Sequence

```
Gen 100: ✅ Improving | Injections: 0
Gen 105: ✅ Improving | Injections: 0
Gen 110: ⚠️ Stagnation: 5 gens | Injections: 0
Gen 115: ⚠️ Stagnation: 10 gens | Injections: 0
Gen 120: ⚠️ Stagnation: 15 gens | Injections: 0
Gen 125: ⚠️ Stagnation: 20 gens | Injections: 0
Gen 130: 🔄 STAGNATION DETECTED (score: 7/10, 25 gens) - Injecting diversity!
Gen 130: ✅ Improving | Injections: 1  (counter reset)
Gen 135: ✅ Improving | Injections: 1
Gen 140: ⚠️ Stagnation: 5 gens | Injections: 1
...
```

## Files Modified

1. `src/deap_optimizer/evolutionary_algorithm.py`:
   - Added counter tracking logic
   - Updated display to show counter
   - Added checkpoint save/load for counter
   - Added improvement detection

2. `STAGNATION_DETECTION.md`:
   - Added counter documentation
   - Added display examples

## Testing

Test the counter with:

```bash
python test_stagnation_detection.py
```

Watch for:
- Counter incrementing when stagnant
- Counter resetting on improvement
- Counter persisting through checkpoints
- Display showing correct values
