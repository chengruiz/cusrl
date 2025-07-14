from pathlib import Path

import torch

__all__ = ["Trial"]


class Trial:
    """A class to manage and load checkpoints.

    This class provides a convenient way to interact with the output of a
    training run (a "trial"). It can be initialized with a path to:
      - An experiment's home directory (e.g., `DummyEnv:DummyAlgo/`)
      - A trial's home directory (e.g., `DummyEnv:DummyAlgo/trial_1/`)
      - A specific checkpoint file (e.g., `DummyEnv:DummyAlgo/trial_1/ckpt/ckpt_100.pt`)

    The class automatically discovers information like the experiment name,
    algorithm, environment, and available checkpoint iterations.

    Args:
        path (str):
            The path to the experiment, trial directory or a specific checkpoint file.
            If an experiment is provided, the latest trial is loaded.
            If a trial directory is provided, the latest checkpoint is loaded.
            Otherwise, the specific checkpoint file is expected.
        verbose (bool, optional):
            If True, prints the path of the loaded checkpoint. Defaults to True.

    Attributes:
        home (Path):
            The absolute path to the trial's root directory.
        name (str):
            The name of the trial, derived from its directory name.
        experiment_name (str | None):
            The name of the experiment, if the directory name follows the 'env:algo' convention.
        algorithm_name (str | None):
            The name of the algorithm, parsed from the experiment name.
        environment_name (str | None):
            The name of the environment, parsed from the experiment name.
        iteration (int):
            The iteration number of the loaded checkpoint.
        checkpoint_path (Path):
            The full path to the selected checkpoint file.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the path points to a file that is not a valid
            checkpoint file (e.g., does not start with 'ckpt_' or end with '.pt').
    """

    home: Path
    name: str
    experiment_name: str | None
    algorithm_name: str | None
    environment_name: str | None
    iteration: int
    checkpoint_path: Path

    def __init__(self, path: str, verbose: bool = True):
        trial_path: Path = Path(path)
        if not trial_path.exists():
            raise FileNotFoundError(f"'{trial_path}' not found.")

        if trial_path.is_dir():
            self.home = trial_path.absolute()
            if (self.home / "latest").is_symlink():
                self.home = (self.home / "latest").resolve()
            self.all_iterations = self._search_ckpt(self.home / "ckpt")
            self.iteration = max(self.all_iterations)
        else:
            if not trial_path.name.startswith("ckpt_") or trial_path.suffix != ".pt":
                raise ValueError(f"'{trial_path}' is not a valid directory or checkpoint file.")

            self.home = trial_path.parent.parent.absolute()
            self.all_iterations = self._search_ckpt(self.home / "ckpt")
            self.iteration = self._get_ckpt_iteration(trial_path)

        self.name = self.home.name
        self.experiment_name = self.home.parent.name
        if self.experiment_name.count(":") == 1:
            self.environment_name, self.algorithm_name = self.experiment_name.split(":")
        else:  # If the naming does not follow the convention
            self.environment_name = self.algorithm_name = self.experiment_name = None
        self.checkpoint_path = self.home / f"ckpt/ckpt_{self.iteration}.pt"

        if verbose:
            print(f"Trial loaded from \033[4m{self.checkpoint_path}\033[0m.")

    def load_checkpoint(self, map_location: str | torch.device | None = None):
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
        return checkpoint

    @classmethod
    def _search_ckpt(cls, directory: Path) -> list[int]:
        ckpt_iterations = []
        for filename in directory.iterdir():
            ckpt_iterations.append(cls._get_ckpt_iteration(filename))
        ckpt_iterations.sort()
        return ckpt_iterations

    @classmethod
    def _get_ckpt_iteration(cls, filename: Path) -> int:
        return int(filename.stem.rsplit("_", 1)[-1])
