"""
Text preprocessing and normalization pipeline for Business Entity Resolution (Phase 2).

Standardizes business names and addresses, strips 35+ international corporate legal suffixes,
expands street and state abbreviations, normalizes Unicode diacritics via NFKD,
and constructs name-boosted composite search text for blocking and feature extraction.
"""

import re
import unicodedata
from typing import Optional, List, Tuple

# Comprehensive Multilingual Corporate Suffix Regex (35+ Suffixes, D-09)
# Ordered from longest to shortest to prevent premature sub-token matching.
LEGAL_SUFFIXES: List[str] = [
    # Multi-word and long forms first
    r'\bentreprise unipersonnelle a responsabilite limitee\b',
    r'\bsociete par actions simplifiee\b',
    r'\bsociete a responsabilite limitee\b',
    r'\blimited liability partnership\b',
    r'\blimited liability company\b',
    r'\bprofessional corporation\b',
    r'\bsociete en nom collectif\b',
    r'\btrading company\b',
    r'\bprivate limited\b',
    r'\bsociete anonyme\b',
    r'\bincorporation\b',
    r'\bincorporated\b',
    r'\bcorporation\b',
    r'\benterprises\b',
    r'\bindustries\b',
    r'\btrading co\b',
    r'\bsolutions\b',
    r'\bservices\b',
    r'\bpvt\s+ltd\b',
    r'\blimited\b',
    r'\bcompany\b',
    r'\beurl\b',
    r'\bsarl\b',
    r'\bgmbh\b',
    r'\bcorp\b',
    r'\bpvt\b',
    r'\bllp\b',
    r'\bllc\b',
    r'\binc\b',
    r'\bltd\b',
    r'\bsas\b',
    r'\bsnc\b',
    r'\bspa\b',
    r'\bco\b',
    r'\bpc\b',
    r'\bsa\b',
    r'\bag\b',
    r'\bbv\b',
    r'\bnv\b',
]

LEGAL_SUFFIX_REGEX = re.compile('|'.join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Common address street / unit abbreviations (D-11)
STREET_ABBREVIATIONS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bhwy\b': 'highway',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpkwy\b': 'parkway',
    r'\bter\b': 'terrace',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor',
    r'\bno\b': 'number',
}

STREET_REGEX_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(k, flags=re.IGNORECASE), v) for k, v in STREET_ABBREVIATIONS.items()
]

# State abbreviations to expand (US & India common) (D-11)
STATE_ABBREVIATIONS = {
    # US states
    r'\bny\b': 'new york',
    r'\bca\b': 'california',
    r'\btx\b': 'texas',
    r'\bfl\b': 'florida',
    r'\bil\b': 'illinois',
    r'\bpa\b': 'pennsylvania',
    r'\boh\b': 'ohio',
    r'\bga\b': 'georgia',
    r'\bnc\b': 'north carolina',
    r'\bmi\b': 'michigan',
    r'\bnj\b': 'new jersey',
    r'\bva\b': 'virginia',
    r'\bwa\b': 'washington',
    r'\baz\b': 'arizona',
    r'\bma\b': 'massachusetts',
    r'\btn\b': 'tennessee',
    r'\bmo\b': 'missouri',
    r'\bmd\b': 'maryland',
    r'\bwi\b': 'wisconsin',
    r'\bmn\b': 'minnesota',
    r'\bco\b': 'colorado',
    r'\bal\b': 'alabama',
    r'\bsc\b': 'south carolina',
    r'\bla\b': 'louisiana',
    # India states
    r'\bup\b': 'uttar pradesh',
    r'\bmp\b': 'madhya pradesh',
    r'\bap\b': 'andhra pradesh',
    r'\bwb\b': 'west bengal',
    r'\bmh\b': 'maharashtra',
    r'\bdl\b': 'delhi',
    r'\bka\b': 'karnataka',
    r'\brj\b': 'rajasthan',
    r'\bgj\b': 'gujarat',
    r'\bkl\b': 'kerala',
    r'\bpb\b': 'punjab',
    r'\bhr\b': 'haryana',
    r'\btn\b': 'tamil nadu',
}

STATE_REGEX_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(k, flags=re.IGNORECASE), v) for k, v in STATE_ABBREVIATIONS.items()
]

# Punctuation cleaning regex preserving unicode alphanumeric characters
PUNCT_REGEX = re.compile(r'[^\w\s]', flags=re.UNICODE)
MULTI_SPACE_REGEX = re.compile(r'\s+')


def strip_accents(text: Optional[str]) -> str:
    """
    Normalize unicode characters and remove diacritic marks (D-10).

    Uses NFKD decomposition to separate characters from combining accents,
    stripping Latin combining diacritics while preserving Indic, Cyrillic, and
    non-Latin characters intact.
    """
    if text is None or not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped or stripped.lower() in ("nan", "null", "none"):
        return ""

    # Replace non-breaking spaces and control whitespaces
    sanitized = stripped.replace('\u00a0', ' ').replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

    # Normalize NFKD to separate characters and combining diacritics
    nfkd_form = unicodedata.normalize('NFKD', sanitized)
    
    chars = []
    for c in nfkd_form:
        if unicodedata.combining(c):
            # Preserve Indic combining marks (Devanagari, Tamil, etc.: U+0900 to U+0D7F)
            if 0x0900 <= ord(c) <= 0x0D7F:
                chars.append(c)
        else:
            chars.append(c)

    return ''.join(chars)


def normalize_business_name(name: Optional[str]) -> str:
    """
    Standardize business names (D-09, D-10):
    - Strips accents/diacritics and converts to lowercase.
    - Replaces '&' with ' and ', '@' with ' at '.
    - Strips 35+ international corporate legal suffixes.
    - Strips punctuation while preserving alphanumeric characters.
    - Collapses whitespace and trims.
    """
    if name is None or not isinstance(name, str):
        return ""

    s = strip_accents(name).lower()
    if not s or s in ("nan", "null", "none"):
        return ""

    s = s.replace('&', ' and ').replace('@', ' at ')
    s = LEGAL_SUFFIX_REGEX.sub(' ', s)
    s = PUNCT_REGEX.sub(' ', s)
    s = MULTI_SPACE_REGEX.sub(' ', s).strip()
    return s


def normalize_business_address(address: Optional[str]) -> str:
    """
    Standardize business addresses (D-10, D-11):
    - Strips accents/diacritics and converts to lowercase.
    - Replaces '&' with ' and ', '@' with ' at '.
    - Expands street, unit, and state abbreviations.
    - Strips punctuation while preserving digits and words.
    - Collapses whitespace and trims.
    """
    if address is None or not isinstance(address, str):
        return ""

    s = strip_accents(address).lower()
    if not s or s in ("nan", "null", "none"):
        return ""

    s = s.replace('&', ' and ').replace('@', ' at ')

    # Expand street abbreviations
    for pattern, replacement in STREET_REGEX_PATTERNS:
        s = pattern.sub(replacement, s)

    # Expand state abbreviations
    for pattern, replacement in STATE_REGEX_PATTERNS:
        s = pattern.sub(replacement, s)

    s = PUNCT_REGEX.sub(' ', s)
    s = MULTI_SPACE_REGEX.sub(' ', s).strip()
    return s


def create_search_text(name: Optional[str], address: Optional[str], name_weight: int = 2) -> str:
    """
    Create a unified search text from normalized name and address (D-12).

    The business name is repeated `name_weight` times (default 2x) to provide
    stronger lexical signal during TF-IDF tokenization and fuzzy matching.
    """
    clean_name = normalize_business_name(name)
    clean_addr = normalize_business_address(address)

    parts: List[str] = []
    if clean_name:
        parts.extend([clean_name] * max(1, name_weight))
    if clean_addr:
        parts.append(clean_addr)

    return ' '.join(parts).strip()


if __name__ == "__main__":
    test_cases = [
        ("Payne Énterprises LLC", "3315 FREMONT ST, PEORIA, IL"),
        ("Ss Food Private Limited", "Af-684, Ghaziabad, UP"),
        ("Maure Wilblims Colombier Inc", None),
        ("Raj Investments LLP", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu"),
        ("Dahlia Power Reliable Scientific LLC", "Missouri, 630 45th Terrace, Kansas City"),
        ("Carrefour Supermarché SAS", "12 Rue de la Paix, Paris"),
        ("Amazon Services LLC & Co.", "123 Main St, Ste 400, NY"),
        ("Reliance Industries Pvt Ltd", "Plot 45, MG Rd, MH")
    ]

    print("Testing Preprocessing Pipeline:")
    for name, addr in test_cases:
        c_name = normalize_business_name(name)
        c_addr = normalize_business_address(addr)
        c_all = create_search_text(name, addr)
        print(f"Original: {name} | {addr}")
        print(f"  Norm Name: '{c_name}'")
        print(f"  Norm Addr: '{c_addr}'")
        print(f"  Search Text: '{c_all}'")
        print("-" * 50)

    assert strip_accents("Café de la Société") == "Cafe de la Societe"
    assert strip_accents(None) == ""
    assert strip_accents("nan") == ""
    assert normalize_business_name("Reliance Industries Pvt Ltd") == "reliance"
    assert normalize_business_name("Carrefour Supermarché SAS") == "carrefour supermarche"
    assert normalize_business_name("Amazon Services LLC & Co.") == "amazon and"
    assert normalize_business_address("123 Main St, Ste 400, NY") == "123 main street suite 400 new york"
    assert normalize_business_address("Plot 45, MG Rd, MH") == "plot 45 mg road maharashtra"
    assert "acme acme 100 broadway street" in create_search_text("Acme Corp", "100 Broadway St")
    print("All preprocessing tests passed successfully!")
