import torch

from PyUnity.ultralytics.ultralytics import YOLO

from SAE.train_main import DSIAC_validator as dsiac_mi # Main Training class!!!!

import mlflow

seed = 124
torch.manual_seed(seed)



DEVICE = "gpu" #"mps" if torch.backends.mps.is_available() else "cpu"
LOAD_MODEL = True

experiment_name = "mech_interp"

def main():
    model = YOLO()
    path = '/Users/nmital/Documents/Work/ARL/ATR Database/cegr/png_files/'

    dsiac_trainer = dsiac_mi(data_path=path, device=DEVICE, nc=11, LOAD_MODEL=LOAD_MODEL, experiment_name=experiment_name)

    dsiac_trainer.train_sae()
    dsiac_trainer.inspect_sae()


if __name__ == "__main__":
    main()