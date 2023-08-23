from neuro_ml.dataset import SimulationEnum, DatasetParams
from neuro_ml.models import (
    EdgeRegressor,
    EdgeRegressorParams,
)
from neuro_ml.fit import fit, test_model
import torch
import sys
from typing_extensions import dataclass_transform
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

def fit_edge_regressor(dataset_params):
    # Set the number of time steps we want to calculate co-firing rates for and the number of neurons
    edge_regressor_params = EdgeRegressorParams(n_shifts=10, n_neurons=dataset_params.n_neurons)

    # Fit the model
    fit(
        EdgeRegressor,
        model_is_classifier=False,
        model_params=edge_regressor_params,
        dataset_params=dataset_params,
        device=device,
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the simulation, window size, and number of files to use
    dataset_params = DatasetParams(
        n_neurons=20,
        n_timesteps=1_000,
        timestep_bin_length=500,
        number_of_files=100,
        simulation_enum=SimulationEnum.mikkel,
    )

    fit_edge_regressor(dataset_params)

    # graph_simple_params = EdgeRegressor(
    # n_shifts = 20
    # )
    # test_model(EdgeRegressor, epoch=10, dataset_params=dataset_params, model_params = graph_simple_params, model_is_classifier=False, device=device)
