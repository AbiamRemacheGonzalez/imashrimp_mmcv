import os
import json
from .hook import HOOKS, Hook


@HOOKS.register_module()
class EarlyStopping(Hook):

    def __init__(self, patience, window=20, early_trial_stop_epoch=None):
        self.max_patience = patience
        self.average_window = window
        self.early_trial_stop_epoch = early_trial_stop_epoch

        self.train_metrics = {}
        self.train_losses = {}
        self.average_train_losses = {}

        self.val_metrics = {}
        self.val_losses = {}
        self.average_val_losses = {}

        self.patience_count = 0

        self.iterations_epoch = {}
        self.iterations_lrs = {}

        self.saving_name = "_training_accumulate_info.json"

    @staticmethod
    def get_epoch(runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    @staticmethod
    def get_iter(runner, inner_iter=False):
        """Get the current training iteration step."""
        if inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    @staticmethod
    def window_average_loss(losses, window=20):
        sum_losses = 0
        count = 0
        for loss in reversed(losses):
            sum_losses = sum_losses + loss
            count += 1
            if count == window:
                break
        return sum_losses / count

    def before_run(self, runner):
        # Load the previous results when the training is recovering
        self.load_training_information(runner)

    def load_training_information(self, runner):
        training_path = os.path.join(runner.work_dir, self.saving_name)
        if os.path.exists(training_path):
            training_information = self._load_dict(training_path)
            if training_information is not None:
                self.train_metrics = training_information.get('train_metrics', {})
                self.train_losses = training_information.get('train_losses', {})
                self.average_train_losses = training_information.get('average_train_losses', {})
                self.val_metrics = training_information.get('val_metrics', {})
                self.val_losses = training_information.get('val_losses', {})
                self.average_val_losses = training_information.get('average_val_losses', {})
                self.iterations_epoch = training_information.get('iterations_epoch', {})
                self.iterations_lrs = training_information.get('iterations_lrs', {})

    @staticmethod
    def _load_dict(path):
        try:
            with open(path, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
            return datos
        except FileNotFoundError:
            print("❌ File not found.")
            return None
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return None

    def after_run(self, runner):
        # Save all the results
        self.save_training_information(runner)

    def save_training_information(self, runner):
        training_information = {
            'train_metrics': self.train_metrics,
            'train_losses': self.train_losses,
            'average_train_losses': self.average_train_losses,
            'val_metrics': self.val_metrics,
            'val_losses': self.val_losses,
            'average_val_losses': self.average_val_losses,
            'iterations_epoch': self.iterations_epoch,
            'iterations_lrs': self.iterations_lrs
        }
        training_path = os.path.join(runner.work_dir, self.saving_name)
        self._save_json_file(training_path, training_information)

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
        pass

    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        output = runner.log_buffer.output

        # Saving epoch training results
        epoch = self.get_epoch(runner)
        self.train_losses[epoch] = output['loss']
        self.average_train_losses[epoch] = self.window_average_loss(list(self.train_losses.values()),
                                                                    self.average_window)
        self.train_metrics[epoch] = output['acc_pose']
        # ----------------------------

        runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        output = runner.log_buffer.output
        # ----------------------------
        # Saving epoch training results
        epoch = self.get_epoch(runner)
        self.val_losses[epoch] = output['loss']
        self.average_val_losses[epoch] = self.window_average_loss(list(self.val_losses.values()), self.average_window)
        self.val_metrics[epoch] = output['acc_pose']

        # ----------------------------
        # Early Stopping Managing
        min_average_val_loss = min(self.average_val_losses.values())
        last_average_val_loss = self.average_val_losses[epoch]

        val_difference = last_average_val_loss - min_average_val_loss
        val_percentual_error = val_difference / min_average_val_loss
        if val_percentual_error > 0.1:
            min_average_train_loss = min(self.average_train_losses.values())
            last_average_train_loss = self.average_train_losses[epoch]

            train_difference = last_average_train_loss - min_average_train_loss
            train_percentual_error = train_difference / min_average_train_loss
            if train_percentual_error > 0.1:
                self.patience_count += 1
                runner.logger.info(f"Patience {self.patience_count} / {self.max_patience}")
                if self.patience_count >= self.max_patience:
                    runner.logger.info(f"Patience reached.")
                    runner.early_stop = True

        # ----------------------------
        # Trial Stop Managing
        if self.early_trial_stop_epoch is not None:
            if epoch >= self.early_trial_stop_epoch:
                runner.logger.info(f"Early trial stop epoch reached.")
                runner.early_stop = True

        # ----------------------------
        min_val_loss = min(self.val_losses.values())
        last_val_loss = self.val_losses[epoch]

        max_val_pck = max(self.val_metrics.values())
        last_val_pck = self.val_metrics[epoch]

        runner.log_buffer.update({'pck_checkpoint_save': max_val_pck == last_val_pck})
        runner.log_buffer.update({'loss_checkpoint_save': min_val_loss == last_val_loss})
        runner.log_buffer.update({'min_val_loss': min_val_loss})
        runner.log_buffer.update({'max_val_pck': max_val_pck})

        self.save_training_information(runner)

    def before_train_iter(self, runner):
        epoch = self.get_epoch(runner)
        iteration = self.get_iter(runner, inner_iter=True)
        self.iterations_epoch[iteration] = epoch
        self.iterations_lrs[iteration] = runner.optimizer.param_groups[0]['lr']
