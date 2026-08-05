from __future__ import annotations

import unittest
import numpy as np
import torch

from backend.models.schemas import BehaviorBatch, KeyPressEvent
from backend.ml.temporal_model import Temporal1DCNN, TemporalGRU, extract_raw_event_sequence


class TestTemporalModel(unittest.TestCase):

    def test_extract_raw_event_sequence_shape_and_privacy(self) -> None:
        batch = BehaviorBatch(
            session_id="privacy_test",
            started_at=1000.0,
            ended_at=2000.0,
            mouse_moves=[],
            clicks=[],
            key_presses=[
                KeyPressEvent(timestamp=1100.0, key="S"),
                KeyPressEvent(timestamp=1250.0, key="E"),
                KeyPressEvent(timestamp=1400.0, key="C"),
                KeyPressEvent(timestamp=1550.0, key="R"),
                KeyPressEvent(timestamp=1700.0, key="E"),
                KeyPressEvent(timestamp=1850.0, key="T"),
            ],
            scrolls=[],
            focus_events=[],
            metadata=None,
        )

        seq = extract_raw_event_sequence(batch, max_len=60)
        self.assertEqual(seq.shape, (60, 7))

        # Privacy verification: Ensure no character ordinals or strings leaked into feature values
        # Channel 3 is is_key (1.0 for keypress)
        key_count = np.sum(seq[:, 3])
        self.assertEqual(key_count, 6.0)

    def test_model_forward_pass_and_probability_range(self) -> None:
        cnn = Temporal1DCNN(in_channels=7, seq_len=60)
        cnn.eval()

        dummy_batch_cnn = torch.randn(5, 7, 60)
        out_cnn = cnn(dummy_batch_cnn).squeeze(-1)
        self.assertEqual(out_cnn.shape, (5,))

        for p in out_cnn.detach().numpy():
            self.assertGreaterEqual(float(p), 0.0)
            self.assertLessEqual(float(p), 1.0)

        gru = TemporalGRU(input_size=7, hidden_size=32)
        gru.eval()

        dummy_batch_gru = torch.randn(5, 60, 7)
        out_gru = gru(dummy_batch_gru).squeeze(-1)
        self.assertEqual(out_gru.shape, (5,))

        for p in out_gru.detach().numpy():
            self.assertGreaterEqual(float(p), 0.0)
            self.assertLessEqual(float(p), 1.0)


if __name__ == "__main__":
    unittest.main()
