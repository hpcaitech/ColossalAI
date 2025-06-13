# Re-import required libraries due to kernel reset
import argparse
from collections import defaultdict
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Process profiling logs and generate a timeline plot.")
parser.add_argument("--visualization", type=str, default="actor_timelines.png", help="Path to the visualization file.")
args = parser.parse_args()

# Raw log lines
log_lines = []

import glob

files = glob.glob("*.prof")
for file in files:
    with open(file, "r") as f:
        log_lines += f.readlines()

# Parse logs and collect function intervals grouped by actor
actors = defaultdict(lambda: defaultdict(list))
current_entries = {}

for line in log_lines:
    if line.startswith("[Log]"):
        continue
    parts = line.split()
    timestamp = float(parts[0])
    actor = parts[1]
    action = parts[3]
    func_name = parts[4]
    key = (actor, func_name)
    if action == "Enter":
        current_entries[key] = timestamp
    elif action == "Exit":
        start_time = current_entries.pop(key, None)
        if start_time is not None:
            actors[actor][func_name].append((start_time, timestamp))

# Plotting setup
fig, ax = plt.subplots(figsize=(12, 6))
colors = cm.get_cmap("tab10", len(actors))

actor_offsets = {}
base_offset = 0
function_spacing = 0.9

yticks = []
yticklabels = []

for idx, (actor, func_dict) in enumerate(actors.items()):
    actor_offsets[actor] = base_offset
    color = colors(idx)
    for j, (func, intervals) in enumerate(func_dict.items()):
        y_val = base_offset + j * function_spacing
        yticks.append(y_val)
        yticklabels.append(f"{actor}:{func}")
        for start, end in intervals:
            ax.plot(
                [datetime.fromtimestamp(start), datetime.fromtimestamp(end)],
                [y_val, y_val],
                color=color,
                linewidth=4,
                label=actor if j == 0 else "",
            )
    base_offset += len(func_dict) * function_spacing + 1

# Formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Time")
ax.set_title("Timeline per Actor")
# Remove duplicate labels in legend
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())
plt.tight_layout()
plt.grid(True)
plt.savefig(args.visualization)
print(f"Plot saved as {args.visualization}")
