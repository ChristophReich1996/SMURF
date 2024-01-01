from typing import List

import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.utils import flow_to_image

from smurf import raft_smurf

def main() -> None:
    # Load images
    image1: Tensor = torchvision.io.read_image("toy_data/reds/00000000.png", mode=torchvision.io.ImageReadMode.RGB)
    image2: Tensor = torchvision.io.read_image("toy_data/reds/00000004.png", mode=torchvision.io.ImageReadMode.RGB)
    # Normalize image to the pixe range of [-1, 1]
    image1 = 2.0 * (image1 / 255.0) - 1.0
    image2 = 2.0 * (image2 / 255.0) - 1.0
    # Init SMURF RAFT model
    model: nn.Module = raft_smurf(checkpoint="smurf_kitti.pt")
    # Predict optical flow
    optical_flow: List[Tensor] = model(image1[None], image2[None])
    # Plot optical flow
    torchvision.io.write_png(flow_to_image(optical_flow[-1][0]), "github/optical_flow.png")


if __name__ == "__main__":
    main()
