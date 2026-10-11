#!/usr/bin/env python3
"""Export tagged notebook definitions to an importable Python module."""

import argparse
import ast
import json
from pathlib import Path


def export_notebook(notebook_path, output_path):
    notebook_path = Path(notebook_path)
    output_path = Path(output_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    exported_statements = []

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        syntax_tree = ast.parse(source)

        for statement in syntax_tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not statement.name.startswith("check_"):
                    exported_statements.append(statement)

            elif (isinstance(statement, ast.If)
                  and isinstance(statement.test, ast.Name)
                  and statement.test.id == "TYPE_CHECKING"):
                exported_statements.append(statement)

            elif isinstance(
                statement,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.ClassDef,
                    ast.Assign,
                    ast.AnnAssign,
                ),
            ):
                exported_statements.append(statement)

    # Future imports must precede all other statements.
    future_imports = []
    other_statements = []

    for statement in exported_statements:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            future_imports.append(statement)
        else:
            other_statements.append(statement)

    body = "\n\n".join(
        ast.unparse(statement) for statement in future_imports + other_statements
    )

    header = (
        "#!/usr/bin/env python3\n"
        f"# Exported from {notebook_path.name}; "
        "edit the notebook, then re-export.\n"
    )

    result = header + "\n" + body + "\n"

    compile(result, str(output_path), "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--output", default="src/slm/slm/core.py")
    arguments = parser.parse_args()
    print(export_notebook(arguments.notebook, arguments.output))


if __name__ == "__main__":
    main()
