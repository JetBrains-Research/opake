from __future__ import annotations

import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_distribution_inventory as inventory  # noqa: E402


class DistributionInventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.wheel_dir = self.root / "dist"
        self.wheel_dir.mkdir()

    def _write_workspace(
        self,
        *,
        members: tuple[tuple[str, str], ...] = (
            ("packages/base", "Example_Base"),
            ("packages/core", "example-core"),
        ),
        exclude: tuple[str, ...] = (),
    ) -> Path:
        member_paths = ", ".join(f'"{path}"' for path, _name in members)
        exclude_paths = ", ".join(f'"{path}"' for path in exclude)
        workspace = f"members = [{member_paths}]\n"
        if exclude:
            workspace += f"exclude = [{exclude_paths}]\n"
        root_pyproject = self.root / "pyproject.toml"
        root_pyproject.write_text(
            '[project]\nname = "Example"\n\n[tool.uv.workspace]\n' + workspace,
            encoding="utf-8",
        )
        for path, name in members:
            member = self.root / path
            member.mkdir(parents=True)
            (member / "pyproject.toml").write_text(
                f'[project]\nname = "{name}"\n', encoding="utf-8"
            )
        return root_pyproject

    def _write_wheel(self, name: str, *, tag: str = "py3-none-any") -> None:
        stem = name.replace("-", "_")
        wheel_path = self.wheel_dir / f"{stem}-1.0-{tag}.whl"
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr(
                f"{stem}-1.0.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
            )

    def test_exact_inventory_collapses_platform_wheels(self) -> None:
        pyproject = self._write_workspace()
        self._write_wheel("Example")
        self._write_wheel("example-base")
        self._write_wheel("example-core", tag="cp311-abi3-manylinux_2_28_x86_64")
        self._write_wheel("example-core", tag="cp311-abi3-macosx_11_0_arm64")

        expected = inventory.expected_distributions(pyproject)
        actual = inventory.built_distributions(self.wheel_dir)

        self.assertEqual(inventory.inventory_errors(expected, actual), [])

    def test_missing_distribution_is_reported(self) -> None:
        pyproject = self._write_workspace()
        self._write_wheel("example")
        self._write_wheel("example-base")

        errors = inventory.inventory_errors(
            inventory.expected_distributions(pyproject),
            inventory.built_distributions(self.wheel_dir),
        )

        self.assertEqual(errors, ["missing distributions: example-core"])

    def test_unexpected_distribution_is_reported(self) -> None:
        pyproject = self._write_workspace()
        for name in ("example", "example-base", "example-core", "stale-wheel"):
            self._write_wheel(name)

        errors = inventory.inventory_errors(
            inventory.expected_distributions(pyproject),
            inventory.built_distributions(self.wheel_dir),
        )

        self.assertEqual(errors, ["unexpected distributions: stale-wheel"])

    def test_duplicate_project_name_is_rejected(self) -> None:
        pyproject = self._write_workspace(
            members=(
                ("packages/one", "same-name"),
                ("packages/two", "same_name"),
            )
        )

        with self.assertRaisesRegex(ValueError, "duplicate distribution name"):
            inventory.expected_distributions(pyproject)

    def test_workspace_excludes_are_honored(self) -> None:
        pyproject = self._write_workspace(
            members=(
                ("packages/base", "example-base"),
                ("packages/private", "example-private"),
            ),
            exclude=("packages/private",),
        )

        self.assertEqual(
            inventory.expected_distributions(pyproject),
            {"example", "example-base"},
        )

    def test_malformed_wheel_is_rejected(self) -> None:
        with zipfile.ZipFile(self.wheel_dir / "broken-1.0-py3-none-any.whl", "w"):
            pass

        with self.assertRaisesRegex(ValueError, "expected one"):
            inventory.built_distributions(self.wheel_dir)


if __name__ == "__main__":
    unittest.main()
