import json
import math

# Generate exponential curves for the limits
limits = {}

# Parameters for exponential curves
# Steps 1-50: Initial exponential growth/decay
# Steps 51-100: Different exponential rates

for step in range(1, 51):
    if step <= 50:
        # First phase (steps 1-50)
        # drawdown_worst: exponential decay from 0.99 to ~0.6
        # gain: exponential growth from 1 to ~10000
        # rsquared: exponential growth from 0.01 to ~0.5
        
        progress = (step - 1) / 49  # 0 to 1
        
        # Exponential decay for drawdown_worst
        drawdown_start = 0.5
        drawdown_end = 0.33
        drawdown_value = drawdown_start * math.exp(progress * math.log(drawdown_end / drawdown_start))
        
        drawdown_1pt_start = 0.4
        drawdown_1pt_end = 0.2
        drawdown_1pt_value = drawdown_start * math.exp(progress * math.log(drawdown_1pt_end / drawdown_1pt_start))
        
        # Exponential growth for gain
        gain_start = 1
        gain_end = 1000
        gain_value = gain_start * math.exp(progress * math.log(gain_end / gain_start))
        
        # Exponential growth for rsquared
        rsquared_start = 0.97
        rsquared_end = 0.9887
        rsquared_value = rsquared_start * math.exp(progress * math.log(rsquared_end / rsquared_start))
        
    # Round values appropriately
    limits[f"drawdown_worst-{step}"] = round(drawdown_value, 4)
    limits[f"drawdown_worst_mean_1pct-{step}"] = round(drawdown_1pt_value, 4)
    limits[f"gain-{step}"] = int(round(gain_value))
    limits[f"rsquared-{step}"] = round(rsquared_value, 4)

# Print the limits in JSON format
print(json.dumps(limits, indent=2))