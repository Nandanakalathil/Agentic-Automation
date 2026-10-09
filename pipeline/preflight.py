"""Deterministic repository scan that runs BEFORE the analyst agent.

Cheap, offline, no LLM: detects build files, pinned CUDA/torch versions, test
setup, weight sources, licence terms and native/bio dependencies by reading a
handful of known paths. The result is handed to the analyst as established fact,
so it spends its steps on the inference path instead of rediscovering that the
repo has a requirements.txt.

Two things here are not just cost savings:

* `licence.redistributable` is False for weights that may not be served (e.g.
  AlphaFold3's terms are non-commercial, by request). The analyst cannot be
  trusted to notice this reliably, and it must be visible at the EC2 approval
  gate - before any money is spent.
* `cuda_pins` feeds the test-fixer's compatibility matrix, turning native build
  failures from trial-and-error into a lookup.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Files that deterministically establish how a project is built.
BUILD_FILES = [
    "requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
    "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "poetry.lock",
    "environment.yml", "environment.yaml", "conda.yml", "conda.yaml",
    "Dockerfile", "docker-compose.yml", "apt.txt", "postBuild", "runtime.txt",
    "Makefile", "install.sh", "setup.sh",
]

TEST_MARKERS = ["pytest.ini", "tox.ini", "noxfile.py", "conftest.py", ".github/workflows"]

# Native / CUDA-sensitive packages: these are what actually break builds.
NATIVE_DEPS = [
    "torch", "torchvision", "torchaudio", "torch-geometric", "torch_geometric",
    "torch-scatter", "torch-sparse", "dgl", "jax", "jaxlib", "tensorflow",
    "deepspeed", "flash-attn", "flash_attn", "xformers", "triton", "apex",
    "mmcv", "detectron2", "bitsandbytes", "vllm",
]

# Enzyme-engineering / synthetic-biology stack. Presence of these means the
# endpoint schema will need FASTA/PDB/A3M types rather than JSON primitives,
# and that native builds are likely (PyRosetta, OpenMM, HH-suite).
BIO_DEPS = [
    "biopython", "Bio", "biotite", "pyrosetta", "openmm", "rdkit", "mdanalysis",
    "prody", "py3dmol", "nglview", "fair-esm", "esm", "alphafold", "colabfold",
    "proteinmpnn", "rfdiffusion", "pdbfixer", "hhsuite", "pyhmmer", "scikit-bio",
]

# Licences under which model weights may NOT simply be redeployed as a service.
# Substring match against the licence text, lowercased.
RESTRICTED_MARKERS = [
    ("alphafold 3", "AlphaFold3 terms: non-commercial, weights by request"),
    ("alphafold3", "AlphaFold3 terms: non-commercial, weights by request"),
    ("noncommercial", "non-commercial clause"),
    ("non-commercial", "non-commercial clause"),
    ("cc-by-nc", "Creative Commons NonCommercial"),
    ("cc by-nc", "Creative Commons NonCommercial"),
    ("research use only", "research-use-only clause"),
    ("academic use only", "academic-use-only clause"),
    ("llama 2 community", "Llama community licence: MAU threshold applies"),
    ("llama 3 community", "Llama community licence: MAU threshold applies"),
    ("openrail", "OpenRAIL use restrictions apply"),
    ("not for commercial", "non-commercial clause"),
]

PERMISSIVE_MARKERS = ["mit license", "apache license", "bsd ", "bsd-3", "bsd-2",
                      "mozilla public license", "isc license", "unlicense"]

# Weight sources worth knowing before planning the instance and the volume.
HF_RE = re.compile(r"""["']([A-Za-z0-9][\w.-]*/[\w.-]+)["']""")
URL_RE = re.compile(r"""https?://[^\s"'<>)]+\.(?:pt|pth|bin|safetensors|tar|tar\.gz|zip|npz|ckpt|h5|pkl|params)""")
WEIGHT_EXT = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5", ".npz", ".pkl", ".params"}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox"}

# Code and weights are frequently licensed differently - Apache-2.0 source with
# non-commercial weights is the norm for large bio models - so they are read
# separately. Reading only the first LICENSE file found hides exactly the term
# that forbids serving the model.
CODE_LICENCE_FILES = ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md",
                      "LICENCE.txt", "COPYING", "COPYING.txt"]
WEIGHTS_LICENCE_FILES = ["LICENSE-WEIGHTS", "LICENSE_WEIGHTS", "WEIGHTS_LICENSE", "MODEL_LICENSE",
                         "LICENSE.model", "LICENSE.weights", "MODEL_TERMS.md", "TERMS.md",
                         "TERMS_OF_USE.md", "WEIGHTS_TERMS.md"]

# Provenance links a reader needs: the paper, the model card, the runnable demo.
LINK_PATTERNS = {
    "arxiv": re.compile(r"https?://arxiv\.org/(?:abs|pdf)/[\w.v/-]+"),
    "biorxiv": re.compile(r"https?://(?:www\.)?biorxiv\.org/content/[\w./-]+"),
    "doi": re.compile(r"https?://(?:dx\.)?doi\.org/[\w./()-]+"),
    "huggingface": re.compile(r"https?://huggingface\.co/[\w.\-/]+"),
    "colab": re.compile(r"https?://colab\.research\.google\.com/[\w.\-/?=]+"),
    "github": re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+"),
    "nature": re.compile(r"https?://(?:www\.)?nature\.com/articles/[\w.-]+"),
}

NOTEBOOK_EXT = {".ipynb"}
CONFIG_EXT = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
# Example/reference inputs a first-time reader can actually run the model on.
SAMPLE_EXT = {".fasta", ".fa", ".faa", ".pdb", ".cif", ".mmcif", ".a3m", ".sto",
              ".csv", ".tsv", ".sdf", ".mol2", ".smi", ".png", ".jpg", ".npy"}
ARCHIVED_MARKERS = ["this repository is archived", "no longer maintained", "deprecated",
                    "unmaintained", "read-only", "superseded by"]

# Version pins that drive native builds, e.g. "torch==2.4.1", "cuda 12.1".
PIN_RE = re.compile(r"^\s*([A-Za-z][\w.\-]*)\s*([=<>!~]=+|==)\s*([\w.*+]+)", re.M)
CUDA_RE = re.compile(r"cu(\d{3})|cuda[ \-_]?(\d+\.\d+)", re.I)


@dataclass
class Licence:
    name: str = ""
    file: str = ""
    redistributable: bool | None = None  # None = unknown
    reason: str = ""

    @property
    def blocking(self) -> bool:
        """True when the licence forbids (or may forbid) serving this as an endpoint."""
        return self.redistributable is False


@dataclass
class Preflight:
    build_files: list[str] = field(default_factory=list)
    python_version: str = ""
    test_markers: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    native_deps: list[str] = field(default_factory=list)
    bio_deps: list[str] = field(default_factory=list)
    cuda_pins: dict[str, str] = field(default_factory=dict)
    hf_ids: list[str] = field(default_factory=list)
    weight_urls: list[str] = field(default_factory=list)
    bundled_weights_mb: float = 0.0
    git_lfs: bool = False
    entrypoints: list[str] = field(default_factory=list)
    licence: Licence = field(default_factory=Licence)          # the CODE licence
    weights_licence: Licence = field(default_factory=Licence)  # often stricter
    # --- orientation for a first-time reader -----------------------------
    repo_name: str = ""
    owner: str = ""
    readme_title: str = ""
    links: dict[str, list[str]] = field(default_factory=dict)
    tree: list[str] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    notebooks: list[str] = field(default_factory=list)
    cli_entrypoints: list[str] = field(default_factory=list)
    sample_inputs: list[str] = field(default_factory=list)
    last_commit: str = ""
    days_since_commit: int | None = None
    archived_hint: str = ""
    broken_scripts: list[dict] = field(default_factory=list)

    @property
    def is_bio(self) -> bool:
        return bool(self.bio_deps)

    @property
    def effective_licence(self) -> Licence:
        """Whichever of code/weights forbids more. Serving needs both to permit it."""
        if self.weights_licence.blocking:
            return self.weights_licence
        if self.licence.blocking:
            return self.licence
        if self.weights_licence.file and self.weights_licence.redistributable is None:
            return self.weights_licence
        return self.licence

    def to_dict(self) -> dict:
        skip = {"licence", "weights_licence"}
        d = {k: v for k, v in self.__dict__.items() if k not in skip}
        d["licence"] = dict(self.licence.__dict__)
        d["weights_licence"] = dict(self.weights_licence.__dict__)
        d["is_bio"] = self.is_bio
        return d


def _read(p: Path, limit: int = 200_000) -> str:
    try:
        return p.read_text(errors="replace")[:limit]
    except OSError:
        return ""


def _classify_licence(text: str, name: str, rel: str) -> Licence:
    low = text.lower()
    for marker, reason in RESTRICTED_MARKERS:
        if marker in low:
            return Licence(name=name, file=rel, redistributable=False, reason=reason)
    for marker in PERMISSIVE_MARKERS:
        if marker in low:
            return Licence(name=name or marker.strip().title(), file=rel,
                           redistributable=True, reason="permissive licence")
    return Licence(name=name, file=rel, redistributable=None,
                   reason="licence text not recognised - review manually")


def _licence_name_from_pyproject(root: Path) -> str:
    pp = root / "pyproject.toml"
    if not pp.exists():
        return ""
    try:
        data = tomllib.loads(_read(pp))
    except (tomllib.TOMLDecodeError, ValueError):
        return ""
    lic = (data.get("project") or {}).get("license")
    if isinstance(lic, str):
        return lic
    if isinstance(lic, dict):
        return lic.get("text") or lic.get("file") or ""
    return ""


def _python_version(root: Path) -> str:
    pv = root / ".python-version"
    if pv.exists():
        return _read(pv).strip().splitlines()[0] if _read(pv).strip() else ""
    pp = root / "pyproject.toml"
    if pp.exists():
        try:
            data = tomllib.loads(_read(pp))
            rq = (data.get("project") or {}).get("requires-python")
            if rq:
                return str(rq)
        except (tomllib.TOMLDecodeError, ValueError):
            pass
    sp = root / "setup.py"
    if sp.exists():
        m = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)", _read(sp))
        if m:
            return m.group(1)
    for env in ("environment.yml", "environment.yaml"):
        p = root / env
        if p.exists():
            m = re.search(r"python\s*[=><]=?\s*([\d.]+)", _read(p))
            if m:
                return m.group(1)
    return ""


def _first_licence(root: Path, candidates: list[str], fallback_name: str = "") -> Licence:
    for cand in candidates:
        p = root / cand
        if p.exists() and p.is_file():
            return _classify_licence(_read(p, 60_000), fallback_name, cand)
    return Licence()


def _git(root: Path, *args: str) -> str:
    """Read-only git query. Returns '' on any failure - preflight never raises."""
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _origin(root: Path) -> tuple[str, str]:
    """(owner, repo_name) from the git remote, e.g. ('deepmind', 'alphafold')."""
    url = _git(root, "remote", "get-url", "origin")
    m = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    return (m.group(1), m.group(2)) if m else ("", root.name)


def _readme(root: Path) -> tuple[str, str]:
    """(filename, text) of the first README found."""
    for cand in ("README.md", "README.rst", "README.txt", "README", "readme.md"):
        p = root / cand
        if p.exists():
            return cand, _read(p, 120_000)
    return "", ""


def _extract_links(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind, rx in LINK_PATTERNS.items():
        hits = sorted({m.rstrip(").,") for m in rx.findall(text)})
        if hits:
            out[kind] = hits[:8]
    return out


def _tree(root: Path, max_depth: int = 2, max_entries: int = 60) -> list[str]:
    """Indented directory listing with per-directory file counts, for the structure diagram."""
    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        if depth >= max_depth:
            dirnames[:] = []
        if len(lines) >= max_entries:
            lines.append("... (truncated)")
            break
        name = "." if rel == Path(".") else str(rel)
        visible = [f for f in filenames if not f.startswith(".")]
        lines.append(f"{'  ' * depth}{name}/  ({len(visible)} files)")
    return lines


def _collect_assets(root: Path) -> tuple[list[dict], list[str], list[str]]:
    """(notable assets, notebooks, sample input files) - what a reader can actually use."""
    assets: list[dict] = []
    notebooks: list[str] = []
    samples: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if SKIP_DIRS & parts or p.name.startswith("."):
            continue
        rel, ext = str(p.relative_to(root)), p.suffix.lower()
        try:
            size_mb = round(p.stat().st_size / 1e6, 2)
        except OSError:
            continue
        if ext in NOTEBOOK_EXT:
            notebooks.append(rel)
            assets.append({"path": rel, "kind": "notebook", "size_mb": size_mb})
        elif ext in WEIGHT_EXT:
            assets.append({"path": rel, "kind": "weights", "size_mb": size_mb})
        elif ext in SAMPLE_EXT and size_mb < 50:
            samples.append(rel)
            assets.append({"path": rel, "kind": "sample", "size_mb": size_mb})
        elif ext in CONFIG_EXT and len(p.relative_to(root).parts) <= 2:
            assets.append({"path": rel, "kind": "config", "size_mb": size_mb})
        elif p.name in ("Dockerfile", "docker-compose.yml"):
            assets.append({"path": rel, "kind": "docker", "size_mb": size_mb})
    assets.sort(key=lambda a: (-a["size_mb"], a["path"]))
    return assets[:60], sorted(notebooks)[:20], sorted(samples)[:20]


def _broken_scripts(root: Path, limit: int = 20) -> list[dict]:
    """Python files that will not even parse.

    Compiling is not running, so this cannot prove a script works - but a
    SyntaxError proves it does not, and that is worth knowing before anyone
    spends an afternoon on the repo. Python-2-only files surface here too,
    which is the same finding in practice: it will not run on this interpreter.
    """
    broken: list[dict] = []
    for p in sorted(root.rglob("*.py")):
        if SKIP_DIRS & set(p.relative_to(root).parts) or len(broken) >= limit:
            continue
        try:
            compile(p.read_text(errors="replace"), str(p), "exec")
        except SyntaxError as e:
            broken.append({"path": str(p.relative_to(root)),
                           "error": f"line {e.lineno}: {e.msg}"})
        except (OSError, ValueError):
            continue
    return broken


def _days_since(date_str: str) -> int | None:
    """Days since an ISO date at the front of `git log` output."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str or "")
    if not m:
        return None
    try:
        last = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return (date.today() - last).days


def _cli_entrypoints(root: Path) -> list[str]:
    """Scripts a user can actually invoke: argparse CLIs plus declared console_scripts."""
    found: list[str] = []
    for p in root.rglob("*.py"):
        if SKIP_DIRS & set(p.relative_to(root).parts) or len(found) >= 30:
            continue
        txt = _read(p, 40_000)
        if "ArgumentParser" in txt or "click.command" in txt or 'if __name__ == "__main__"' in txt:
            found.append(str(p.relative_to(root)))
    for name in ("setup.py", "setup.cfg", "pyproject.toml"):
        txt = _read(root / name, 40_000) if (root / name).exists() else ""
        found += [f"{e.strip()} (console_script)" for e in re.findall(r"^\s*([\w.-]+\s*=\s*[\w.:]+)\s*$",
                                                                      txt, re.M)
                  if ":" in e][:10]
    return sorted(set(found))[:30]


def scan(workspace: Path) -> Preflight:
    """Read known paths under `workspace` and return established facts. Never raises."""
    root = Path(workspace).resolve()
    pf = Preflight()

    # --- build files + dependency text ------------------------------------
    dep_text = ""
    for name in BUILD_FILES:
        p = root / name
        if p.exists():
            pf.build_files.append(name)
            if p.suffix in {".txt", ".toml", ".yml", ".yaml", ".cfg"} or name in {"Dockerfile", "setup.py"}:
                dep_text += "\n" + _read(p)

    pf.python_version = _python_version(root)

    low_dep = dep_text.lower()
    pf.native_deps = sorted({d for d in NATIVE_DEPS if d.lower() in low_dep})
    pf.bio_deps = sorted({d for d in BIO_DEPS if re.search(rf"\b{re.escape(d.lower())}\b", low_dep)})

    # --- version pins that drive native builds ----------------------------
    for pkg, _op, ver in PIN_RE.findall(dep_text):
        if pkg.lower() in {d.lower() for d in NATIVE_DEPS}:
            pf.cuda_pins[pkg] = ver
    cuda = CUDA_RE.search(dep_text)
    if cuda:
        pf.cuda_pins["cuda"] = cuda.group(1) and f"{cuda.group(1)[:2]}.{cuda.group(1)[2:]}" or cuda.group(2)

    # --- tests ------------------------------------------------------------
    for marker in TEST_MARKERS:
        if (root / marker).exists():
            pf.test_markers.append(marker)
    for d in ("tests", "test"):
        if (root / d).is_dir():
            pf.test_dirs.append(d)

    # --- weights ----------------------------------------------------------
    pf.git_lfs = (root / ".gitattributes").exists() and "filter=lfs" in _read(root / ".gitattributes")
    scripts = [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".md"}
               and not SKIP_DIRS & set(p.relative_to(root).parts)][:400]
    hf, urls = set(), set()
    for p in scripts:
        txt = _read(p, 60_000)
        if "from_pretrained" in txt or "hf_hub_download" in txt or "snapshot_download" in txt:
            hf.update(m for m in HF_RE.findall(txt) if "/" in m and not m.startswith(("./", "../")))
        urls.update(URL_RE.findall(txt))
    pf.hf_ids = sorted(hf)[:20]
    pf.weight_urls = sorted(urls)[:20]

    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_EXT and not SKIP_DIRS & set(p.relative_to(root).parts):
            try:
                total += p.stat().st_size
            except OSError:
                continue
    pf.bundled_weights_mb = round(total / 1e6, 1)

    # --- entrypoints ------------------------------------------------------
    for cand in ("app.py", "main.py", "serve.py", "predict.py", "inference.py",
                 "run.py", "cli.py", "demo.py", "api.py"):
        if (root / cand).exists():
            pf.entrypoints.append(cand)

    # --- licences (code and weights are separate questions) ---------------
    declared = _licence_name_from_pyproject(root)
    pf.licence = _first_licence(root, CODE_LICENCE_FILES, declared)
    if not pf.licence.file:
        pf.licence = (_classify_licence(declared, declared, "pyproject.toml") if declared
                      else Licence(reason="no LICENSE file found - treat as all-rights-reserved until checked"))
    pf.weights_licence = _first_licence(root, WEIGHTS_LICENCE_FILES)
    if not pf.weights_licence.file and (pf.hf_ids or pf.weight_urls):
        pf.weights_licence = Licence(
            reason="weights are downloaded from an external source with no licence file in the repo - "
                   "check the model card's terms before serving")

    # --- orientation ------------------------------------------------------
    pf.owner, pf.repo_name = _origin(root)
    readme_name, readme = _readme(root)
    if readme:
        heading = next((l.lstrip("# ").strip() for l in readme.splitlines() if l.startswith("# ")), "")
        pf.readme_title = heading[:200]
        pf.links = _extract_links(readme)
        low = readme[:4000].lower()
        pf.archived_hint = next((m for m in ARCHIVED_MARKERS if m in low), "")
    pf.last_commit = _git(root, "log", "-1", "--format=%cs %h %s")[:200]
    pf.days_since_commit = _days_since(pf.last_commit)
    pf.tree = _tree(root)
    pf.assets, pf.notebooks, pf.sample_inputs = _collect_assets(root)
    pf.cli_entrypoints = _cli_entrypoints(root)
    pf.broken_scripts = _broken_scripts(root)
    return pf


# ----------------------------------------------------------------- rendering
def to_prompt_block(pf: Preflight) -> str:
    """Established facts for the analyst's task, so it does not spend steps rediscovering them."""
    def lst(xs: list[str], empty: str = "none found") -> str:
        return ", ".join(xs) if xs else empty

    lines = [
        "## Established facts (deterministic scan - trust these, do not re-verify)",
        f"- Build files: {lst(pf.build_files)}",
        f"- Python version constraint: {pf.python_version or 'not declared'}",
        f"- Test markers: {lst(pf.test_markers)}; test dirs: {lst(pf.test_dirs)}",
        f"- Native/CUDA-sensitive deps: {lst(pf.native_deps)}",
        f"- Version pins: {json.dumps(pf.cuda_pins) if pf.cuda_pins else 'none found'}",
        f"- HuggingFace ids referenced: {lst(pf.hf_ids)}",
        f"- Weight URLs referenced: {lst(pf.weight_urls)}",
        f"- Weights bundled in repo: {pf.bundled_weights_mb} MB; git-lfs: {pf.git_lfs}",
        f"- Candidate entrypoints: {lst(pf.entrypoints)}",
        f"- Runnable CLI scripts: {lst(pf.cli_entrypoints[:15])}",
        f"- Notebooks: {lst(pf.notebooks)}",
        f"- Example input files in repo: {lst(pf.sample_inputs)}",
        f"- Code licence: {pf.licence.name or 'unknown'} ({pf.licence.file or 'no file'}) - {pf.licence.reason}",
        f"- Weights licence: {pf.weights_licence.name or 'unknown'} "
        f"({pf.weights_licence.file or 'no separate file'}) - {pf.weights_licence.reason or 'not stated'}",
        f"- Origin: {pf.owner or '?'}/{pf.repo_name or '?'}; last commit: {pf.last_commit or 'unknown'}",
        f"- README title: {pf.readme_title or 'none'}",
        f"- Links found in README: {json.dumps(pf.links) if pf.links else 'none'}",
        f"- Maintenance signal: {pf.archived_hint or 'none found'}"
        + (f"; last commit was {pf.days_since_commit} days ago" if pf.days_since_commit is not None else ""),
        f"- Python files that DO NOT PARSE: "
        + (", ".join(f"{b['path']} ({b['error']})" for b in pf.broken_scripts[:10])
           if pf.broken_scripts else "none"),
        "",
        "Repository structure (directories, file counts):",
        *(f"    {line}" for line in pf.tree[:40]),
    ]
    if pf.is_bio:
        lines += [
            f"- Bio/chem stack detected: {lst(pf.bio_deps)}",
            "  This is a computational-biology repository. Expect FASTA/PDB/mmCIF/A3M inputs rather than",
            "  JSON primitives, residue-level outputs (pLDDT, coordinates), and native builds",
            "  (PyRosetta/OpenMM/HH-suite) that need system packages. Record the exact I/O formats.",
        ]
    lines.append("\nSpend your steps on the inference path, I/O formats and hardware - not on the above.")
    return "\n".join(lines)


def _one_licence_warning(lic: Licence, what: str) -> str:
    if lic.redistributable is True or (not lic.file and not lic.reason):
        return ""
    head = "LICENCE BLOCKER" if lic.blocking else "LICENCE UNVERIFIED"
    return (f"{head} ({what}): {lic.name or 'unknown licence'} ({lic.file or 'no licence file'}) - {lic.reason}. "
            "Deploying this model as an endpoint may not be permitted. Confirm the terms before approving.")


def licence_warning(pf: Preflight) -> str:
    """Approval-gate text when either licence may forbid serving. Empty when both are fine.

    Code and weights are asked separately: permissive source with non-commercial
    weights is common, and it is the weights term that stops you serving it.
    """
    parts = [w for w in (_one_licence_warning(pf.licence, "code"),
                         _one_licence_warning(pf.weights_licence, "weights")) if w]
    return "\n".join(parts)
