import unittest
from unittest.mock import patch

import numpy as np
import torch

from gromo.utils.disk_dataset import (
    DiskDataset,
    MemMapDataset,
    SimpleMemMapDataset,
)


class TestDiskDataset(unittest.TestCase):
    def test_iterates_and_concatenates_keys(self):
        """Test that DiskDataset correctly iterates and concatenates input/target keys"""
        in_a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        in_b = torch.tensor([[10.0], [20.0], [30.0]])
        out_y = torch.tensor([[0.0], [1.0], [2.0]])

        def fake_torch_load(path, *args, **kwargs):
            if path == "in.pt":
                return {"a": in_a, "b": in_b}
            if path == "out.pt":
                return {"y": out_y}
            raise FileNotFoundError(path)

        with patch("torch.load", side_effect=fake_torch_load):
            ds = DiskDataset(
                input_filename="in.pt",
                target_filename="out.pt",
                input_keys=["a", "b"],
                target_keys=["y"],
            )
            items = list(ds)
            self.assertEqual(len(items), 3)
            xs, ys = zip(*items)
            for i in range(len(items)):
                expected_x = torch.cat([in_a[i], in_b[i]])
                self.assertTrue(torch.allclose(xs[i], expected_x))
                self.assertTrue(torch.allclose(ys[i], out_y[i]))

        # Mismatched lengths raise error
        in_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with patch("torch.load", side_effect=fake_torch_load):
            ds = DiskDataset(
                input_filename="in.pt",
                target_filename="out.pt",
                input_keys=["a"],
                target_keys=["y"],
            )
            with self.assertRaises((ValueError, RuntimeError)):
                list(ds)


class TestMemMapDataset(unittest.TestCase):
    def test_getitem_returns_tensors(self):
        """Test that MemMapDataset returns torch tensors via __getitem__"""
        input_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        target_np = np.array([[0.0], [1.0]], dtype=np.float32)

        def fake_convert(self, path, keys, name):
            return "__in.npy" if name == "__input" else "__out.npy"

        def fake_np_load(fname, mmap_mode=None):
            if fname == "__in.npy":
                return input_np
            if fname == "__out.npy":
                return target_np
            raise FileNotFoundError(fname)

        with (
            patch.object(
                MemMapDataset, "_MemMapDataset__convert_dict_to_npy", fake_convert
            ),
            patch("numpy.load", side_effect=fake_np_load),
        ):
            ds = MemMapDataset(
                input_filename="in.pt",
                target_filename="out.pt",
                input_keys=["a"],
                target_keys=["y"],
            )
            self.assertEqual(len(ds), 2)
            for i in range(len(ds)):
                x0, y0 = ds[i]
                self.assertIsInstance(x0, torch.Tensor)
                self.assertIsInstance(y0, torch.Tensor)
                self.assertTrue(torch.allclose(x0, torch.from_numpy(input_np[i])))
                self.assertTrue(torch.allclose(y0, torch.from_numpy(target_np[i])))

        # Mismatched lengths raise error
        target_np = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)

        with (
            patch.object(
                MemMapDataset, "_MemMapDataset__convert_dict_to_npy", fake_convert
            ),
            patch("numpy.load", side_effect=fake_np_load),
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                ds = MemMapDataset(
                    input_filename="in.pt",
                    target_filename="out.pt",
                    input_keys=["a"],
                    target_keys=["y"],
                )


class TestSimpleMemMapDataset(unittest.TestCase):
    def test_getitem_concatenates_multiple_files(self):
        """Test that SimpleMemMapDataset correctly concatenates multiple input/target files"""
        in1 = np.array([[1.0], [2.0]], dtype=np.float32)
        in2 = np.array([[10.0], [20.0]], dtype=np.float32)
        out1 = np.array([[0.0], [1.0]], dtype=np.float32)
        out2 = np.array([[100.0], [200.0]], dtype=np.float32)

        fake_map = {
            "in1.npy": in1,
            "in2.npy": in2,
            "out1.npy": out1,
            "out2.npy": out2,
        }

        def fake_np_load(fname, mmap_mode=None):
            if fname not in fake_map:
                raise FileNotFoundError(fname)
            return fake_map[fname]

        with patch("numpy.load", side_effect=fake_np_load):
            ds = SimpleMemMapDataset(
                input_filenames=["in1.npy", "in2.npy"],
                target_filenames=["out1.npy", "out2.npy"],
            )
            self.assertEqual(len(ds), 2)
            for i in range(len(ds)):
                x, y = ds[i]
                self.assertIsInstance(x, torch.Tensor)
                expected_x = np.concatenate([in1[i], in2[i]], axis=0)
                self.assertTrue(np.allclose(x.numpy(), expected_x))
                expected_y = np.concatenate([out1[i], out2[i]], axis=0)
                self.assertTrue(np.allclose(y.numpy(), expected_y))

        # Mismatched lengths raise error
        out1 = np.array([[0.0]], dtype=np.float32)
        fake_map["out1.npy"] = out1

        with patch("numpy.load", side_effect=fake_np_load):
            with self.assertRaises(ValueError):
                ds = SimpleMemMapDataset(
                    input_filenames=["in1.npy", "in2.npy"],
                    target_filenames=["out1.npy", "out2.npy"],
                )

        # Mismatched lengths raise error
        in2 = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        fake_map["in2.npy"] = in2

        with patch("numpy.load", side_effect=fake_np_load):
            with self.assertRaises(ValueError):
                ds = SimpleMemMapDataset(
                    input_filenames=["in1.npy", "in2.npy"],
                    target_filenames=["out1.npy", "out2.npy"],
                )


if __name__ == "__main__":
    unittest.main()
