"""Pack Allrun exports into the small, self-contained unit-test fixture archive."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
output = root.parents[2] / "tests/data/fixtures/ensight_cavity.zip"
output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for prefix in ("", "dual", "parallel"):
        folder = root / prefix / "postProcessing"
        if not folder.is_dir():
            raise FileNotFoundError(f"Run Allrun first: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and (
                path.suffix == ".case" or "data" in path.relative_to(folder).parts
            ):
                archive.write(path, path.relative_to(root))
print(output)
