import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gromo.config import loader


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_load_config(self) -> None:
        root = Path().cwd()

        config, method = loader.load_config()

        path_toml = root / "pyproject.toml"

        if path_toml.is_file():
            tomlfile = loader._load_toml(path_toml)
            if tomlfile.get("tool", {}).get("gromo") is None:
                self.assertEqual(method, "gromo.config")
            else:
                self.assertEqual(method, "pyproject.toml")
        else:
            self.assertEqual(method, "gromo.config")
        self.assertIsInstance(config, dict)

    # def test_tomlfile(self) -> None:
    #     with TemporaryDirectory() as cwd:
    #         root = Path(cwd)


if __name__ == "__main__":
    unittest.main()
