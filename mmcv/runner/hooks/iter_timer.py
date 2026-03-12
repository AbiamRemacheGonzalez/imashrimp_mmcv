# Copyright (c) Open-MMLab. All rights reserved.
import os
import time
import json
import torch
from .hook import HOOKS, Hook
import torch.distributed as dist


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_run(self, runner):
        self.tt = time.time()

    def after_run(self, runner):
        total_time = time.time() - self.tt
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        # Final metrics logging
        output = runner.log_buffer.val_history
        max_val_pck = output.get('max_val_pck', 0)[0]
        min_val_loss = output.get('min_val_loss', 0)[0]
        # -----------------------------
        runner.logger.info(f'Total training time: {hours}h {minutes}m {seconds}s')
        path = os.path.join(runner.work_dir, "_training_results.json")
        data = {
                "training_time": f"{hours}h {minutes}m {seconds}s",
                "training_learning_rate": runner.optimizer.param_groups[0]['initial_lr'] * 8 / 1,
                "training_batch_size": runner.data_loader.batch_size,
                "training_epochs": runner._max_epochs,
                "training_max_memory_mb": self._get_max_memory(runner),
                "max_val_pck": max_val_pck,
                "min_val_loss": min_val_loss
        }
        self._save_json_file(path, data)

    @staticmethod
    def _get_max_memory(runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    @staticmethod
    def _save_json_file(path, data):
        """
        Saves data directly to a JSON file, overwriting any existing content.

        Args:
            path (str): Full path of the JSON file (e.g., 'results/metrics.json').
            data (dict, list, etc.): Data to be saved.
        """
        # 1. Ensure the parent directory exists before saving
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 2. Save the data directly, overwriting if the file exists
        with open(path, 'w', encoding='utf-8') as f:
            # indent=4 makes the JSON human-readable
            # ensure_ascii=False allows saving non-ASCII characters properly
            json.dump(data, f, indent=4, ensure_ascii=False)

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
