from dataclasses import dataclass

@dataclass
class DatasetParams:
    n_neurons: int
    n_timesteps: int
    number_of_files: int

    @property
    def foldername(self):
        return f"{self.n_neurons}_neurons_{self.n_timesteps}_timesteps"
