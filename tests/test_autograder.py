"""
Autograder for CSE 150A Practice Homework 1: Hangman + Bayesian Networks.

Strategy:
  1. Read the student's Jupyter notebook (CSE150A_Hangman_HW1.ipynb).
  2. Execute every code cell into a single shared namespace.
  3. Heuristically locate two inference functions in that namespace:
       - a "random" guesser
       - a "Bayesian" guesser
     Both follow the engine's callable signature:
       f(letters_tried: set[str], word_pattern: list[str],
         word_counts: dict[str, int]) -> str
  4. Run targeted correctness tests and a benchmark accuracy test.

We are deliberately tolerant about function *names* because the notebook
documents behavior rather than enforcing names. We identify functions by
probing their behavior on controlled inputs.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Callable

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "CSE150A_Hangman_HW1.ipynb"
WORD_COUNTS_PATH = REPO_ROOT / "hw1_word_counts_05.txt"
HANGMAN_MODULE_PATH = REPO_ROOT / "hangman.py"

# ---------------------------------------------------------------------------
# Notebook loading
# ---------------------------------------------------------------------------


def _load_notebook_namespace() -> dict:
    """Execute all code cells of the student notebook into a fresh namespace.

    Lines that look like IPython magics (starting with '!' or '%') are
    stripped so the script runs outside a Jupyter kernel. Cells that raise
    are reported but do not stop the rest from executing, because students
    often leave exploratory cells around the implementation.
    """
    import nbformat

    if not NOTEBOOK_PATH.exists():
        pytest.fail(
            f"Could not find notebook at {NOTEBOOK_PATH}. "
            "Make sure CSE150A_Hangman_HW1.ipynb is at the repo root."
        )

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    # Expose the hangman engine as a module, since students do `from hangman import ...`
    sys.path.insert(0, str(REPO_ROOT))

    ns: dict = {"__name__": "__student_notebook__"}

    # Patterns on a line that mean "skip this line": interactive demo calls,
    # Colab-only imports, shell/magic commands. We want to extract FUNCTION
    # DEFINITIONS from the notebook, not execute every demo the student left
    # in the cells.
    SKIP_LINE_SUBSTRINGS = (
        "hangman_game(",   # interactive game demo - reads stdin, hangs tests
        "google.colab",    # Colab-only imports
        "drive.mount",     # Colab drive mount
        "files.upload",    # Colab file upload
        "input(",          # any raw_input prompt
    )

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source_lines = []
        for line in cell.source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                continue  # drop shell / magic commands
            if any(s in line for s in SKIP_LINE_SUBSTRINGS):
                # Replace with a pass-through so indentation stays valid
                # (in case the line was inside a function or block).
                source_lines.append(" " * (len(line) - len(stripped)) + "pass  # skipped by autograder")
                continue
            source_lines.append(line)
        source = "\n".join(source_lines)
        if not source.strip():
            continue
        try:
            exec(compile(source, f"<notebook cell {idx}>", "exec"), ns)
        except Exception as e:
            # Don't fail the test suite just because a demo cell errored;
            # we only care whether the required functions got defined.
            print(f"[autograder] warning: cell {idx} raised {type(e).__name__}: {e}")
    return ns


def _load_word_counts() -> dict[str, int]:
    word_counts: dict[str, int] = {}
    with open(WORD_COUNTS_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            word, count = parts
            word_counts[word] = int(count)
    return word_counts


# ---------------------------------------------------------------------------
# Candidate function discovery
# ---------------------------------------------------------------------------


def _callable_inference_candidates(ns: dict) -> list[tuple[str, Callable]]:
    """Return (name, fn) pairs for callables that look like inference fns.

    An "inference function" accepts exactly three positional arguments:
    (letters_tried, word_pattern, word_counts). We test by calling each
    candidate with a benign input and checking it returns a single letter.
    """
    candidates: list[tuple[str, Callable]] = []
    probe_letters_tried: set[str] = set()
    probe_pattern: list[str] = ["_", "_", "_", "_", "_"]
    probe_counts = {"APPLE": 10, "ABOUT": 5, "OTHER": 3}

    for name, obj in ns.items():
        if name.startswith("_"):
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        # Exclude things imported from the engine
        if getattr(obj, "__module__", None) in {"hangman", "builtins", "random"}:
            continue
        try:
            result = obj(set(probe_letters_tried), list(probe_pattern), dict(probe_counts))
        except TypeError:
            continue  # wrong arity
        except Exception:
            continue  # raises on empty state; not a valid candidate
        if isinstance(result, str) and len(result) == 1 and result.isalpha():
            candidates.append((name, obj))
    return candidates


def _classify_candidates(
    candidates: list[tuple[str, Callable]],
) -> tuple[Callable | None, Callable | None]:
    """Split candidates into (random_fn, bayesian_fn).

    Detection by behavior, with name hints as a tiebreaker:
      - Random: returns many distinct letters over many calls with no evidence.
      - Bayesian: on a diagnostic state where one letter is uniquely optimal,
        it returns that letter.
    """
    if not candidates:
        return None, None

    # Name-based hints
    random_hints = ("random", "uniform", "rand")
    bayes_hints = ("bayes", "bayesian", "infer", "posterior", "predict", "smart", "optimal")

    by_name_random = [(n, f) for n, f in candidates if any(h in n.lower() for h in random_hints)]
    by_name_bayes = [(n, f) for n, f in candidates if any(h in n.lower() for h in bayes_hints)]

    # Behavioral probe
    wc = _load_word_counts() if WORD_COUNTS_PATH.exists() else None

    def looks_random(fn: Callable) -> bool:
        rng_state = random.getstate()
        try:
            outs = set()
            for _ in range(40):
                letters_tried: set[str] = set()
                pattern = ["_"] * 5
                guess = fn(letters_tried, list(pattern), dict(wc or {"APPLE": 1}))
                outs.add(guess.upper())
            # A uniform-random guesser should produce many distinct letters.
            return len(outs) >= 10
        except Exception:
            return False
        finally:
            random.setstate(rng_state)

    def looks_bayesian(fn: Callable) -> bool:
        """Call fn on probe states and compare against the exact argmax.

        We compute the ground-truth best letter by enumerating compatible
        words in the corpus. A correct Bayesian implementation must match
        the argmax (within a small tolerance for ties). A random guesser
        will match by chance only ~1/26 of the time per probe.
        """
        if wc is None:
            return False
        probes = [
            (set(), ["_"] * 5),
            ({"A", "E"}, ["_"] * 5),
            ({"Z", "Q", "X"}, ["_"] * 5),
        ]
        matches = 0
        for letters_tried, pattern in probes:
            try:
                guess = fn(set(letters_tried), list(pattern), dict(wc))
            except Exception:
                return False
            if not isinstance(guess, str) or len(guess) != 1 or not guess.isalpha():
                return False
            g = guess.upper()
            if g in letters_tried:
                return False
            # Compute ground-truth argmax for this probe state.
            compat = [(w, c) for w, c in wc.items()
                      if all((p == "_" or w[i] == p) and (p != "_" or w[i] not in letters_tried)
                             for i, p in enumerate(pattern))]
            if not compat:
                continue
            total = sum(c for _, c in compat)
            blanks = [i for i, p in enumerate(pattern) if p == "_"]
            scores = {}
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if letter in letters_tried:
                    continue
                s = 0.0
                for w, c in compat:
                    if any(w[i] == letter for i in blanks):
                        s += c / total
                scores[letter] = s
            if not scores:
                continue
            max_score = max(scores.values())
            if scores.get(g, -1.0) >= max_score - 1e-6:
                matches += 1
        return matches >= 2  # allow 1 probe to fail for robustness

    random_fn = None
    bayes_fn = None

    # Prefer name-matched picks, then fall back to behavior.
    for name, fn in by_name_random:
        if looks_random(fn):
            random_fn = fn
            break
    for name, fn in by_name_bayes:
        if looks_bayesian(fn):
            bayes_fn = fn
            break

    if random_fn is None:
        for _, fn in candidates:
            if fn is bayes_fn:
                continue
            if looks_random(fn):
                random_fn = fn
                break
    if bayes_fn is None:
        for _, fn in candidates:
            if fn is random_fn:
                continue
            if looks_bayesian(fn):
                bayes_fn = fn
                break

    return random_fn, bayes_fn


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ns() -> dict:
    return _load_notebook_namespace()


@pytest.fixture(scope="module")
def word_counts() -> dict[str, int]:
    if not WORD_COUNTS_PATH.exists():
        pytest.fail(
            f"Could not find word counts file at {WORD_COUNTS_PATH}. "
            "Make sure hw1_word_counts_05.txt is at the repo root."
        )
    return _load_word_counts()


@pytest.fixture(scope="module")
def inference_fns(ns):
    candidates = _callable_inference_candidates(ns)
    if not candidates:
        pytest.fail(
            "Could not find any inference functions in the notebook. "
            "Your inference function must accept (letters_tried, word_pattern, "
            "word_counts) and return a single uppercase letter."
        )
    random_fn, bayes_fn = _classify_candidates(candidates)
    return {"random": random_fn, "bayes": bayes_fn, "all": candidates}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_random_guess(inference_fns, word_counts):
    """Problem (b): uniformly random, never repeats a tried letter."""
    fn = inference_fns["random"]
    if fn is None:
        pytest.fail(
            "No random-guess function detected. It should return a uniformly "
            "random letter from the alphabet that has not been guessed yet."
        )

    ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # 1. Never returns a previously-tried letter.
    for seed in range(25):
        random.seed(seed)
        tried = {"A", "E", "I", "O", "U", "S", "T", "R", "N", "L"}
        pattern = ["_"] * 5
        guess = fn(set(tried), list(pattern), dict(word_counts))
        assert isinstance(guess, str), f"Expected str, got {type(guess).__name__}"
        assert len(guess) == 1 and guess.isalpha(), f"Expected single letter, got {guess!r}"
        g = guess.upper()
        assert g in ALPHABET, f"Guess {g!r} is not a letter A-Z"
        assert g not in tried, f"Guess {g!r} was already tried"

    # 2. Over many calls with empty state the distribution covers most of A-Z.
    seen: set[str] = set()
    for i in range(400):
        random.seed(i)
        guess = fn(set(), ["_"] * 5, dict(word_counts)).upper()
        seen.add(guess)
    assert len(seen) >= 20, (
        f"Random guesser only produced {len(seen)} distinct letters over 400 "
        "trials; it does not appear to be uniformly random over A-Z."
    )


def test_bayesian_inference(inference_fns, word_counts):
    """Problem (c): pick the letter that maximizes P(letter in word | evidence)."""
    fn = inference_fns["bayes"]
    if fn is None:
        pytest.fail(
            "No Bayesian inference function detected. It should compute "
            "P(L_i = letter for some i | evidence) and return the argmax letter."
        )

    def compatible(word: str, tried: set[str], pattern: list[str]) -> bool:
        """Is `word` consistent with the current evidence?"""
        for i, p in enumerate(pattern):
            if p != "_" and word[i] != p:
                return False
            if p == "_" and word[i] in tried:
                return False
        return True

    def brute_force_best(tried: set[str], pattern: list[str]) -> tuple[str, dict[str, float]]:
        """Compute the ground-truth best next letter by exhaustive enumeration."""
        compat_words = [(w, c) for w, c in word_counts.items() if compatible(w, tried, pattern)]
        assert compat_words, "No compatible words - bad test state"
        total = sum(c for _, c in compat_words)
        # P(letter in unguessed position | evidence)
        scores: dict[str, float] = {}
        blank_positions = [i for i, p in enumerate(pattern) if p == "_"]
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter in tried:
                continue
            s = 0.0
            for w, c in compat_words:
                if any(w[i] == letter for i in blank_positions):
                    s += c / total
            scores[letter] = s
        best_letter = max(scores, key=scores.get)
        return best_letter, scores

    # A handful of diagnostic states; the student's guess must match the
    # argmax, with small margin for ties.
    probes = [
        (set(), ["_"] * 5),
        ({"A", "I"}, ["_"] * 5),
        ({"D", "I", "M"}, ["M", "_", "D", "_", "M"]),  # the README's worked example
        ({"E"}, ["_", "_", "_", "_", "_"]),
        ({"S", "T", "A", "R"}, ["_"] * 5),
    ]

    for tried, pattern in probes:
        guess = fn(set(tried), list(pattern), dict(word_counts))
        assert isinstance(guess, str) and len(guess) == 1 and guess.isalpha(), (
            f"Bayesian inference returned {guess!r}, expected a single letter"
        )
        g = guess.upper()
        assert g not in tried, f"Bayesian guess {g!r} was already tried"

        best, scores = brute_force_best(tried, pattern)
        # Accept any letter within 1e-6 of the maximum score (ties).
        max_score = scores[best]
        assert scores.get(g, -1) >= max_score - 1e-6, (
            f"State tried={sorted(tried)}, pattern={pattern}: "
            f"expected argmax letter {best!r} (P={max_score:.4f}), "
            f"got {g!r} (P={scores.get(g, 0):.4f})"
        )


def test_benchmark_accuracy(inference_fns, word_counts):
    """Problem (d): Bayesian policy should reach ~93%+ accuracy over many games."""
    fn = inference_fns["bayes"]
    if fn is None:
        pytest.fail("No Bayesian inference function detected; cannot benchmark.")

    # We re-implement a non-interactive game loop so the autograder doesn't
    # depend on clear_output / time.sleep from hangman.py.
    def play_one(seed: int) -> int:
        rng = random.Random(seed)
        word = rng.choices(
            list(word_counts.keys()),
            weights=list(word_counts.values()),
            k=1,
        )[0]
        pattern = ["_"] * len(word)
        tried: set[str] = set()
        misses = 0
        max_misses = 6
        while misses < max_misses and "_" in pattern:
            guess = fn(set(tried), list(pattern), dict(word_counts))
            if not isinstance(guess, str) or len(guess) != 1 or not guess.isalpha():
                return 0
            g = guess.upper()
            if g in tried:
                return 0
            tried.add(g)
            if g in word:
                for i, ch in enumerate(word):
                    if ch == g:
                        pattern[i] = g
            else:
                misses += 1
        return 1 if "_" not in pattern else 0

    N_GAMES = 500  # smaller than 1000 to stay within 15s CI timeout
    wins = sum(play_one(seed) for seed in range(N_GAMES))
    accuracy = wins / N_GAMES
    print(f"Bayesian accuracy over {N_GAMES} games: {accuracy:.3f}")
    assert accuracy >= 0.90, (
        f"Bayesian policy won only {wins}/{N_GAMES} = {accuracy:.1%} of games; "
        "expected at least 90% (assignment target is ~93%+)."
    )
