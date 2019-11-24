from typing import List, Tuple

import numpy as np  # type: ignore
import pyro  # type: ignore
import pyro.contrib.gp as gp  # type: ignore
import torch
from pyro.infer import SVI, Trace_ELBO  # type: ignore
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:1")


class GPLoader(Dataset):
    def __init__(self, years: List[str], device: torch.device) -> None:
        self.x = torch.Tensor()
        self.y = torch.Tensor()
        self.device = device

        for year in years:
            x_arr = np.load(f"./x_{year}.npy")
            y_arr = np.load(f"./y_{year}.npy")

            self.x = torch.cat((self.x, torch.from_numpy(x_arr)))
            self.y = torch.cat((self.y, torch.from_numpy(y_arr)))

            for i in range(12, 24):
                max_ = self.x[:, i].max()
                self.x[:, i] = self.x[:, i] / max_

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i].to(self.device), self.y[i].to(self.device)

    def get_inducing_points(self, n: int) -> torch.Tensor:
        return self.x[:n, :]


def train() -> None:
    dataset = GPLoader(["2017"], device)

    kernel = gp.kernels.Matern32(input_dim=707)
    gpr = gp.models.GPRegression(
        dataset.x.to(device), dataset.y.squeeze(1).to(device), kernel
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": 5})
    svi = SVI(gpr.model, gpr.guide, optimizer, loss=Trace_ELBO())

    for i in range(200):
        loss = svi.step()
        print(f"iter: {i}, loss: {loss}")

    test_set = GPLoader(["2017"], device)
    dataloader = DataLoader(test_set, batch_size=1)

    for i, (x, y) in enumerate(dataloader):
        mu, sigma = gpr(x, full_cov=True)
        print(f"pred: {mu.item()}, label: {y.item()}, error: {mu.item() - y.item()}")

    print(gpr.kernel.variance.item())
    print(gpr.kernel.lengthscale.item())
    print(gpr.noise.item())


if __name__ == "__main__":
    train()
