from __future__ import annotations

import json
from pathlib import Path


def load_turkish_names(path: Path) -> dict:
    """Load Turkish name patterns from a JSON file.

    Expected keys: ``given_names`` (list), ``surname_suffixes`` (list),
    ``diacritic_chars`` (string).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "given_names": list(data.get("given_names", [])),
        "surname_suffixes": list(data.get("surname_suffixes", [])),
        "diacritic_chars": str(data.get("diacritic_chars", "")),
    }


def default_turkish_names() -> dict:
    """Return a hard-coded fallback list of Turkish name patterns.

    Contains ~200 common Turkish given names, common surname suffixes, and the
    Turkish-specific diacritic characters. This keeps the pipeline runnable even
    when ``data/turkish_names.json`` is missing.
    """
    given_names = [
        "ahmet",
        "mehmet",
        "mustafa",
        "ali",
        "hasan",
        "hüseyin",
        "ibrahim",
        "murat",
        "osman",
        "kemal",
        "yusuf",
        "ömer",
        "ramazan",
        "halil",
        "süleyman",
        "abdullah",
        "mahmut",
        "recep",
        "fatih",
        "can",
        "deniz",
        "emre",
        "kerem",
        "tolga",
        "serkan",
        "onur",
        "burak",
        "batuhan",
        "arda",
        "barış",
        "berkay",
        "cenk",
        "çağlar",
        "çağrı",
        "cem",
        "cemal",
        "cengiz",
        "cüneyt",
        "doğan",
        "doğukan",
        "ece",
        "ege",
        "ekrem",
        "elçin",
        "emir",
        "engin",
        "enes",
        "erdem",
        "erdi",
        "erdinç",
        "ergün",
        "erhan",
        "erkan",
        "ertan",
        "ertuğrul",
        "fahri",
        "faruk",
        "ferhat",
        "fırat",
        "fuat",
        "gökay",
        "gökberk",
        "gökhan",
        "görkem",
        "güçlü",
        "gürkan",
        "hakan",
        "haluk",
        "hamza",
        "harun",
        "hatice",
        "hikmet",
        "hilmi",
        "hülya",
        "ilker",
        "ipek",
        "irfan",
        "ışık",
        "ışıl",
        "jale",
        "kaan",
        "kadir",
        "kamer",
        "kayhan",
        "kazım",
        "kenan",
        "koray",
        "kürsat",
        "levent",
        "mert",
        "metin",
        "mücahit",
        "mümtaz",
        "nadir",
        "nazım",
        "necati",
        "necmettin",
        "niyazi",
        "nurettin",
        "oğuz",
        "oğuzhan",
        "okan",
        "olcay",
        "önder",
        "orkun",
        "özgür",
        "özkan",
        "pınar",
        "rami",
        "rıdvan",
        "rıza",
        "sabri",
        "sadık",
        "safa",
        "salih",
        "samet",
        "sami",
        "savaş",
        "seçkin",
        "sedat",
        "selçuk",
        "selim",
        "semih",
        "serdar",
        "serhat",
        "sezgin",
        "sinan",
        "şükrü",
        "tamer",
        "tayfun",
        "taylan",
        "tayyip",
        "timur",
        "tolgahan",
        "tufan",
        "tuna",
        "tuncay",
        "turgay",
        "turgut",
        "ufuk",
        "uğur",
        "umut",
        "utku",
        "yağız",
        "yakup",
        "yalçın",
        "yavuz",
        "yekta",
        "yerlikaya",
        "yiğit",
        "yılmaz",
        "yunus",
        "zafer",
        "ziya",
        "zeynep",
        "ayşe",
        "fatma",
        "emine",
        "hatice",
        "meryem",
        "şerife",
        "sultan",
        "hanife",
        "hacer",
        "fidan",
        "pelin",
        "sevgi",
        "yasemin",
        "nur",
        "selma",
        "büşra",
        "esra",
        "özlem",
        "seda",
        "elif",
        "nilay",
        "gizem",
        "merve",
        "aslı",
        "dilek",
        "hande",
        "tuğçe",
        "aylin",
        "selin",
        "ecem",
        "beyza",
        "damla",
        "melike",
        "rabia",
        "sude",
        "yaren",
        "azra",
        "irem",
        "miray",
        "nisa",
        "sena",
        "ebrar",
        "defne",
        "derya",
        "esma",
        "feyza",
        "gülsüm",
    ]
    surname_suffixes = [
        "oğlu",
        "oğullar",
        "zade",
        "gil",
        "lı",
        "li",
        "lu",
        "lü",
    ]
    diacritic_chars = "ı ş ç ğ ü ö İ Ş Ç Ğ Ü Ö"
    return {
        "given_names": given_names,
        "surname_suffixes": surname_suffixes,
        "diacritic_chars": diacritic_chars,
    }


def contains_turkish_diacritic(text: str, chars: str) -> bool:
    """Return True if any Turkish-specific diacritic character appears in ``text``."""
    if not text or not chars:
        return False
    return any(ch in text for ch in chars if ch != " ")


def matches_turkish_name(text: str, names: dict) -> bool:
    """Return True if ``text`` contains a Turkish given name or surname suffix.

    Matching is case-insensitive. For given names we require a whole-word match
    to avoid false positives (e.g. "ali" inside "calibrate"). Surname suffixes
    are matched against the end of the text.
    """
    if not text:
        return False

    lowered = text.lower()

    for name in names.get("given_names", []):
        needle = name.lower()
        idx = lowered.find(needle)
        if idx == -1:
            continue
        # whole-word check: boundaries are start/end of string or non-letter chars
        before_ok = idx == 0 or not lowered[idx - 1].isalpha()
        after_ok = (idx + len(needle) == len(lowered)) or not lowered[idx + len(needle)].isalpha()
        if before_ok and after_ok:
            return True

    for suffix in names.get("surname_suffixes", []):
        if lowered.endswith(suffix.lower()):
            return True

    return False
