from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_logs(log_file_path):
    logs_by_actor = defaultdict(list)

    with open(log_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            timestamp = float(parts[0])
            actor = parts[1]
            event = parts[3]
            function_name = " ".join(parts[4:])
            logs_by_actor[actor].append((timestamp, event, function_name))

    return logs_by_actor


def build_intervals(logs_by_actor):
    actor_intervals = defaultdict(list)

    for actor, events in logs_by_actor.items():
        func_stack = {}
        for timestamp, event, func in events:
            (actor, func)
            if event == "Enter":
                func_stack[func] = timestamp
            elif event == "Exit" and func in func_stack:
                start = func_stack.pop(func)
                actor_intervals[actor].append((func, start, timestamp))

    return actor_intervals


def plot_actor_timelines(actor_intervals):
    fig, ax = plt.subplots(figsize=(12, 6))
    ytick_labels = []
    yticks = []
    color_map = plt.get_cmap("tab10")
    color_lookup = {}

    y = 0
    for idx, (actor, intervals) in enumerate(sorted(actor_intervals.items())):
        color_lookup[actor] = color_map(idx % 10)
        for func, start, end in intervals:
            ax.barh(y, end - start, left=start, height=0.3, color=color_lookup[actor], label=actor)
            ax.text(start, y + 0.1, func, fontsize=8, color="black")
        yticks.append(y)
        ytick_labels.append(actor)
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Unix Timestamp")
    ax.set_title("Ray Actor Function Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Unique legend
    handles = [mpatches.Patch(color=color_lookup[a], label=a) for a in color_lookup]
    ax.legend(handles=handles, title="Actors", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


# ==== Usage ====
# Replace with your actual log file path
import glob

files = glob.glob("*.prof")
logs = {}
for file in files:
    print(f"Processing file: {file}")
    logs_by_actor = parse_logs(log_file_path)
    logs.update(logs_by_actor)
actor_intervals = build_intervals(logs)
plot_actor_timelines(actor_intervals)
