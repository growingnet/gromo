from pathlib import Path
from typing import Any, Optional, Sequence, Union

import tomli


def _load_toml(path: Union[Path, str]) -> dict[str, Any]:
    """Parse toml file

    Parameters
    ----------
    path : Union[Path, str]
        file path

    Returns
    -------
    dict[str, Any]
        file content
    """
    with open(path, "rb") as f:
        return tomli.load(f)


def _find_project_root(srcs: Sequence[str]) -> tuple[Path, str]:
    """Return a directory containing .git or pyproject.toml.

    That directory will be a common parent of all files and directories
    passed in `srcs`.

    If no directory in the tree contains a marker that would specify it's the
    project root, the root of the file system is returned.

    Parameters
    ----------
    srcs : Sequence[str]
        starting directory

    Returns
    -------
    tuple[Path, str]
        project root path and discovery method
    """

    if not srcs:
        srcs = [str(Path.cwd().resolve())]

    path_srcs = [Path(Path.cwd(), src).resolve() for src in srcs]

    # A list of lists of parents for each 'src'. 'src' is included as a
    # "parent" of itself if it is a directory
    src_parents = [
        list(path.parents) + ([path] if path.is_dir() else []) for path in path_srcs
    ]

    common_base = max(
        set.intersection(*(set(parents) for parents in src_parents)),
        key=lambda path: path.parts,
    )

    for directory in (common_base, *common_base.parents):
        if (directory / ".git").exists():
            return directory, ".git directory"

        if (directory / "pyproject.toml").is_file():
            return directory, "pyproject.toml"

    return directory, "file system root"


def _find_pyproject_toml(path_project_root: Path) -> Optional[str]:
    """Find the absolute filepath to a pyproject.toml if it exists

    Parameters
    ----------
    path_project_root : Path
        project root path

    Returns
    -------
    Optional[str]
        path to pyproject.toml file
    """
    path_pyproject_toml = path_project_root / "pyproject.toml"
    if path_pyproject_toml.is_file():
        return str(path_pyproject_toml)
    return None


def _find_project_config(path_project_root: Path) -> dict:
    path_config_file = path_project_root / "gromo.config"

    if path_config_file.is_file():
        return _load_toml(path_config_file)
    return {}


def load_config() -> tuple[dict, str]:
    """Load configuration file for gromo
    If pyproject.toml exists it will override gromo.config

    Returns
    -------
    tuple[dict, str]
        configurations and config file used
    """
    # Get project root folder
    path_project_root, _ = _find_project_root((str(Path.cwd()),))

    # Find pyproject.toml file
    pyproject_toml_path = _find_pyproject_toml(path_project_root)

    if pyproject_toml_path is not None:
        raw_config = _load_toml(pyproject_toml_path).get("tool", {})
        if raw_config.get("gromo") is None:
            raw_config = _find_project_config(path_project_root)
            method = "gromo.config"
        else:
            method = "pyproject.toml"
    else:
        raw_config = _find_project_config(path_project_root)
        method = "gromo.config"

    gromo_config = raw_config.get("gromo", {})

    # Get the environment and corresponding section
    env = gromo_config.get("env", "production")
    config_data = gromo_config.get(env, {})

    return config_data, method
