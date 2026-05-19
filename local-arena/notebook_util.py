"""
Utility to extract submission.py from a Kaggle notebook (.ipynb).
Used by arena.py and watch.py to directly test notebooks.
"""

import json
import os
import tempfile
import hashlib
import re

BUILTIN_AGENT_ALIASES = {"starter", "random"}


def is_builtin_agent(path):
    return isinstance(path, str) and path in BUILTIN_AGENT_ALIASES


def extract_agent_from_notebook(notebook_path):
    """Extract the submission.py code from a Kaggle notebook.
    
    Handles multiple common formats:
      1. %%writefile submission.py / %%writefile main.py
      2. code = r'''...'''  (string-wrapped agent)
      3. Direct agent code with def agent()
    
    Returns path to a temporary .py file.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        # Format 1: %%writefile submission.py / main.py
        for prefix in ("%%writefile submission.py", "%%writefile main.py"):
            if source.startswith(prefix):
                code = source[len(prefix):].lstrip("\n")
                code_parts.append(code)
                break

    if not code_parts:
        # Format 2: code = r'''...''' (string-wrapped agent)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            # Match patterns like: code = r'''..''' or CODE = """..."""
            for pattern_start in ("code = r'''", 'code = r"""', "code = '''", 'code = """'):
                if pattern_start in source.lower()[:50]:
                    # Find the actual start
                    idx = source.lower().find(pattern_start[:8])
                    if idx == -1:
                        continue
                    rest = source[idx:]
                    # Find the delimiter
                    delim_start = rest.find("'''")
                    if delim_start == -1:
                        delim_start = rest.find('"""')
                    if delim_start == -1:
                        continue
                    delim = rest[delim_start:delim_start+3]
                    code_start = delim_start + 3
                    code_end = rest.find(delim, code_start)
                    if code_end == -1:
                        # Take everything after the opening delimiter
                        code_parts.append(rest[code_start:])
                    else:
                        code_parts.append(rest[code_start:code_end])
                    break

    if not code_parts:
        # Format 2b: agent_source = r'''...''' (string-wrapped agent package)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            match = re.search(
                r"(?is)\bagent_source\b\s*=\s*(r?)(['\"]{3})(.*?)\2",
                source,
            )
            if match:
                code_parts.append(match.group(3))
                break

    if not code_parts:
        # Format 3: Direct agent code — find cells containing 'def agent('
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if "def agent(" in source and not source.startswith("%%"):
                code_parts.append(source)

    if not code_parts:
        raise ValueError(
            f"Could not find agent code in {notebook_path}.\n"
            f"Expected: %%writefile submission.py, code = r'''...''', or def agent()."
        )

    full_code = "\n".join(code_parts)

    # Write to a temp file with a stable name (based on notebook path hash)
    # so repeated calls reuse the same file
    nb_hash = hashlib.md5(os.path.abspath(notebook_path).encode()).hexdigest()[:8]
    nb_name = os.path.splitext(os.path.basename(notebook_path))[0]
    
    # Put extracted files in same directory as the notebook
    out_dir = os.path.dirname(os.path.abspath(notebook_path))
    out_path = os.path.join(out_dir, f".extracted_{nb_name}_{nb_hash}.py")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    return out_path


def resolve_agent_path(path):
    """Resolve an agent path - handles both .py and .ipynb files.
    
    Returns (resolved_path, display_name).
    """
    if is_builtin_agent(path):
        return path, path

    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent file not found: {path}")

    if path.endswith(".ipynb"):
        display_name = os.path.splitext(os.path.basename(path))[0]
        extracted = extract_agent_from_notebook(path)
        print(f"  Extracted {display_name}.ipynb -> {os.path.basename(extracted)}")
        return extracted, display_name
    else:
        display_name = os.path.basename(path).replace(".py", "")
        return path, display_name


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python notebook_util.py <notebook.ipynb>")
        sys.exit(1)
    
    path, name = resolve_agent_path(sys.argv[1])
    print(f"Agent: {name}")
    print(f"Path: {path}")
