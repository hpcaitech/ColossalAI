import atexit
import json
import os
import shutil
import time
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

cmap = list(mcolors.TABLEAU_COLORS.values())

LOG_FOLDER = "record.log"
MAX_WAIT_TIME = 20


class Event:
    def __init__(self, start: int, end: int, name: str, rank: int) -> None:
        self.start = start
        self.end = end
        self.name = name
        self.rank = rank


class Recorder:
    def __init__(self) -> None:
        self.rank_to_history: Dict[int, List[Event]] = {}
        self.base_time = time.time()
        self.temp_event = None

        self.export_format = "png"
        self.export_name = "test"
        self.dpi = 500
        self.theme = "dark_background"
        self.figure_width = 30
        self.figure_height = 10
        self.legend_fontsize = 16
        self.device_fontsize = 20
        self.bar_height = 0.2

        if not os.path.exists(LOG_FOLDER):
            os.makedirs(LOG_FOLDER)

    def start(self, name: str, rank: int):
        # TODO : add lock to prevent conflict
        torch.cuda.synchronize()
        start_time = time.time()
        self.temp_event = Event(start_time, None, name, rank)

    def end(self):
        assert self.temp_event is not None, "`start` before `end`"
        torch.cuda.synchronize()
        end_time = time.time()
        self.temp_event.end = end_time
        rank = self.temp_event.rank
        if rank not in self.rank_to_history:
            self.rank_to_history[rank] = []
        self.rank_to_history[rank].append(self.temp_event)
        self.temp_event = None

    def get_history(self):
        return self.history

    def __call__(self, name: str, rank: str):
        self.temp_name = name
        self.temp_rank = rank
        return self

    def __enter__(self):
        name = self.temp_name
        rank = self.temp_rank
        self.start(name, rank)

    def __exit__(self, *args):
        self.end()

    def dump_record(self):
        rank = dist.get_rank()
        rank_to_history = self.rank_to_history
        records = {"base_time": self.base_time, "content": {}}
        for record_rank in rank_to_history:
            history = rank_to_history[record_rank]
            recs = []
            for event in history:
                rec = {"start": event.start, "end": event.end, "name": event.name}
                recs.append(rec)
            records["content"][record_rank] = recs

        dump_name = f"{rank}.json"
        dump_path = os.path.join(LOG_FOLDER, dump_name)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    def merge_recode(self):
        base_time = self.base_time
        world_size = dist.get_world_size()

        wait_time = 0
        while True:
            time.sleep(0.1)
            log_num = len(os.listdir(LOG_FOLDER))
            if log_num == world_size:
                break

            wait_time += 1
            if wait_time >= MAX_WAIT_TIME:
                break

        # merge
        logs_path = [os.path.join(LOG_FOLDER, file) for file in os.listdir(LOG_FOLDER)]
        recoders = {}
        for path in logs_path:
            with open(path, "r", encoding="utf-8") as f:
                recs = json.load(f)
            for record_rank in recs["content"]:
                history = recs["content"][record_rank]
                recoders[record_rank] = []
                for rec in history:
                    recoders[record_rank].append(
                        {"start": rec["start"] - base_time, "end": rec["end"] - base_time, "name": rec["name"]}
                    )

        shutil.rmtree(LOG_FOLDER)
        with open(self.export_name + ".json", "w", encoding="utf-8") as f:
            json.dump(recoders, f, ensure_ascii=False)

    def visualize_record(self):
        with open(self.export_name + ".json", "r", encoding="utf-8") as f:
            records = json.load(f)
        records = dict(records)
        ranks = list(sorted(records.keys()))

        name_list = {}
        plots = {}
        plt.figure(dpi=self.dpi, figsize=[self.figure_width, self.figure_height])
        plt.style.use(self.theme)

        for rank in ranks:
            rank_records = records[rank]
            for rec in rank_records:
                s = rec["start"]
                e = rec["end"]
                name = rec["name"]
                if name not in name_list:
                    name_list[name] = len(name_list)
                bar = plt.barh(rank, width=e - s, height=self.bar_height, left=s, color=cmap[name_list[name]])
                if name not in plots:
                    plots[name] = bar

        plt.legend(list(plots.values()), list(plots.keys()), loc="upper left", fontsize=self.legend_fontsize)
        plt.yticks(ticks=ranks, labels=[f"Device:{rank}" for rank in ranks], fontsize=self.device_fontsize)
        plt.grid(axis="x")
        plt.savefig("{}.{}".format(self.export_name, self.export_format))

    def exit_worker(self):
        if len(self.rank_to_history) == 0:
            return
        self.dump_record()
        # if this is rank 0, wait for merge
        rank = dist.get_rank()

        if rank == 1:
            # take the base time of rank 0 as standard
            self.merge_recode()
            self.visualize_record()


recorder = Recorder()
atexit.register(recorder.exit_worker)
