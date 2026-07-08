import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EPOCHS
from seed import set_global_seed
from train.trainer import Trainer
from utils.visualization import plot_convergence


def main():
    set_global_seed(42)
    trainer = Trainer()
    trainer.fit(epochs=EPOCHS, save_every=10, checkpoint_prefix="full")
    plot_convergence(trainer.losses, trainer.losses)


if __name__ == "__main__":
    main()
