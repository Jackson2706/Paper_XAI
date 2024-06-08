import unittest
import torch
from torch import nn
from losses import ContrastiveLoss, PreferenceComparisonLoss  # Assume these are saved in loss_functions.py

class TestLossFunctions(unittest.TestCase):

    def test_contrastive_loss_zero_distance(self):
        # Setup
        output1 = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        output2 = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        label = torch.tensor([0], dtype=torch.float32)
        criterion = ContrastiveLoss(margin=2.0)

        # Compute loss
        loss = criterion(output1, output2, label)

        # Check if the loss is zero when the outputs are the same and label is 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_contrastive_loss_nonzero_distance(self):
        # Setup
        output1 = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
        output2 = torch.tensor([[3.0, 4.0], [4.0, 5.0]], requires_grad=True)
        label = torch.tensor([1], dtype=torch.float32)
        criterion = ContrastiveLoss(margin=2.0)

        # Compute loss
        loss = criterion(output1, output2, label)

        # Expected cosine similarity
        cosine_similarity = 1- nn.functional.cosine_similarity(output1, output2)
        expected_loss = torch.mean((1 - label) * pow(cosine_similarity, 2) +
                             label * pow(torch.clamp(2.0 - cosine_similarity, min=0.0), 2))

        # Check if the computed loss matches the expected loss
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

if __name__ == '__main__':
    unittest.main()
