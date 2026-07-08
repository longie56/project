from config import DEVICE
from eval.evaluator import evaluate


class StaticTester:
    def __init__(self, model, dataset, decode_type="greedy", n_pomo=None, device=DEVICE, augmentation_enable=False):
        self.model = model
        self.dataset = dataset
        self.decode_type = decode_type
        self.n_pomo = n_pomo
        self.device = device
        self.augmentation_enable = augmentation_enable

    def run(self):
        return evaluate(
            self.model,
            self.dataset,
            decode_type=self.decode_type,
            n_pomo=self.n_pomo,
            device=self.device,
            augmentation_enable=self.augmentation_enable,
        )
