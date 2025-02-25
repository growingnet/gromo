import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gromo.config import loader


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path().cwd()
        self.toml_name = "pyproject.toml"
        self.config_name = "gromo.config"
        self.path_toml = self.root / self.toml_name
        self.path_config = self.root / self.config_name

    def test_find_project_root(self) -> None:
        project_root_path, method = loader._find_project_root((str(self.root),))
        self.assertEqual(self.root, project_root_path)
        self.assertTrue(
            os.path.samefile(self.path_toml, project_root_path / self.toml_name)
        )
        with TemporaryDirectory() as cwd:
            project_root_path, method = loader._find_project_root((str(cwd),))
            self.assertEqual(method, "file system root")
            self.assertEqual(str(project_root_path), "/")
            self.assertFalse((project_root_path / self.toml_name).is_file())

    def test_find_pyproject_toml(self) -> None:
        project_root_path, _ = loader._find_project_root((str(self.root),))

        pyproject_path = loader._find_pyproject_toml(project_root_path)
        self.assertIsNotNone(pyproject_path)
        self.assertEqual(pyproject_path, str(self.path_toml))
        self.assertTrue(os.path.samefile(Path(pyproject_path), self.path_toml))
        with TemporaryDirectory() as cwd:
            pyproject_path = loader._find_pyproject_toml(Path(cwd))
            self.assertIsNone(pyproject_path)

    def test_find_project_config(self) -> None:
        project_root_path, _ = loader._find_project_root((str(self.root),))

        config = loader._find_project_config(project_root_path)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)

        if self.path_config.is_file():
            real_config = loader._load_toml(self.path_config)
            self.assertEqual(config, real_config)
        else:
            self.assertEqual(config, {})

        with TemporaryDirectory() as cwd:
            config = loader._find_project_config(Path(cwd))
            self.assertIsNotNone(config)
            self.assertIsInstance(config, dict)
            self.assertEqual(config, {})

    def test_load_config(self) -> None:
        config, method = loader.load_config()

        if self.path_toml.is_file():
            tomlfile = loader._load_toml(self.path_toml)
            if tomlfile.get("tool", {}).get("gromo") is None:
                self.assertEqual(method, self.config_name)
            else:
                self.assertEqual(method, self.toml_name)
        else:
            self.assertEqual(method, self.config_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)

        with patch("os.getcwd", return_value="/mocked/path"):  # avoid bug of PR #70
            config, method = loader.load_config()
            self.assertIsNotNone(config)
            self.assertIsInstance(config, dict)
            self.assertEqual(config, {})
            self.assertEqual(method, self.config_name)


if __name__ == "__main__":
    unittest.main()
