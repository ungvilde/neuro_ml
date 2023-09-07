import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy import signal # for computing correlation


class AbstractDataset(Dataset):
    def __init__(
        self,
        filenames,
        dataset_params,
    ) -> None:
        super().__init__()

        self._load_x_and_y(
            filenames,
            dataset_params,
        )

    def _load_x_and_y(
        self,
        filenames,
        dataset_params,
    ):
        """
        Load the dataset
        """
        self.X = []
        self.y = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):

            # TODO:
            # - Load each weight and time serier data set (iterate through the different networks)
            # - Load a dense X matrix directly, and the weight matrix
            # - Convert X into a correlation data set 
            # - finally, append to self.X
            # - unsure what they do int terms of slicing X??
            # Convert sparse X to dense
            raw_data = np.load(filename, allow_pickle=True)

            X = raw_data["X"]
            cross_corr = np.empty((n_neurons*(n_neurons-1), n_shifts))

            lags = signal.correlation_lags(n_timesteps, n_timesteps, mode="full")

            for i, j in edges: #TODO: edges should contain all nonordered pairs of two 
                correlation = signal.correlate(X[i], X[j], mode="full")
                cross_corr[i] = correlation[(lags > 0 )*(lags <= n_shifts)] # correlate on non-zero shifts, smaller than largest shift
                cross_corr[j] = correlation[(lags < 0 )*(lags >= -n_shifts)] # correlate on non-zero shifts, smaller than largest shift

            y = (
                torch.tensor(raw_data["W_0"])
            )

            self.X.append(cross_corr.float())
            self.y.append(y.float())

    def _create_edge_indices(self, n_neurons):
        """
        For each simulation in the dataset create an edge index based on the non-zero elements of W_0
        """
        self.edge_index = []

        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            y = y.reshape(n_neurons, n_neurons)
            edge_index = torch.nonzero(y)

            self.edge_index.append(edge_index.T)

    def _create_fully_connected_edge_index(self, n_neurons):
        """
        For each simulation in the dataset create a fully connected edge index
        """
        self.edge_index = []
        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            edge_index = torch.ones(n_neurons, n_neurons)
            self.edge_index.append(edge_index.nonzero().T)

    def create_geometric_data(self):
        """
        Create a list of torch_geometric.data.Data objects from the dataset
        """
        data = []
        for i in range(len(self)):
            inputs, y = self[i].values()
            data_i = Data(inputs["X"], inputs["edge_index"], y=y)
            data.append(data_i)
        return data