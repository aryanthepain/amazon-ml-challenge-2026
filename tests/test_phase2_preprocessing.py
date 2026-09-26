"""
Pytest unit test suite for Phase 2 text preprocessing and normalization (PREP-01, PREP-02, PREP-03).
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (
    strip_accents,
    normalize_business_name,
    normalize_business_address,
    create_search_text,
    LEGAL_SUFFIXES,
    LEGAL_SUFFIX_REGEX,
    STREET_ABBREVIATIONS,
    STATE_ABBREVIATIONS,
)


def test_strip_accents_multilingual_and_edge_cases():
    """Verify NFKD diacritic removal and script preservation (D-10, PREP-01)."""
    # French accents
    assert strip_accents("Café de la Société Électrique") == "Cafe de la Societe Electrique"
    assert strip_accents("Château Crème Brûlée") == "Chateau Creme Brulee"
    assert strip_accents("façade naïve") == "facade naive"

    # German umlauts and Spanish tildes
    assert strip_accents("Müller & Söhne Über") == "Muller & Sohne Uber"
    assert strip_accents("Año Niño") == "Ano Nino"

    # Indic scripts (Hindi, Tamil transliterated / native) preserved intact
    assert strip_accents("नमस्त") == "नमस्त"
    assert strip_accents("வணக்கம்") == "வணக்கம்"

    # Edge cases: Nulls, string representations of nulls, whitespace
    assert strip_accents(None) == ""
    assert strip_accents("") == ""
    assert strip_accents("   ") == ""
    assert strip_accents("nan") == ""
    assert strip_accents("NAN") == ""
    assert strip_accents("null") == ""
    assert strip_accents("None") == ""


def test_normalize_business_name_suffixes():
    """Verify stripping of 35+ corporate legal suffixes across jurisdictions (D-09, PREP-02)."""
    # US / UK / International
    assert normalize_business_name("Acme Corporation") == "acme"
    assert normalize_business_name("Acme Corp") == "acme"
    assert normalize_business_name("Beta Incorporated") == "beta"
    assert normalize_business_name("Beta Inc.") == "beta"
    assert normalize_business_name("Gamma Limited") == "gamma"
    assert normalize_business_name("Gamma Ltd.") == "gamma"
    assert normalize_business_name("Delta Company") == "delta"
    assert normalize_business_name("Delta Co.") == "delta"
    assert normalize_business_name("Epsilon LLC") == "epsilon"
    assert normalize_business_name("Epsilon Limited Liability Company") == "epsilon"
    assert normalize_business_name("Zeta LLP") == "zeta"
    assert normalize_business_name("Zeta Limited Liability Partnership") == "zeta"
    assert normalize_business_name("Theta PC") == "theta"
    assert normalize_business_name("Theta Professional Corporation") == "theta"

    # India
    assert normalize_business_name("Reliance Industries Pvt Ltd") == "reliance"
    assert normalize_business_name("Tata Services Private Limited") == "tata"
    assert normalize_business_name("Infosys Solutions") == "infosys"
    assert normalize_business_name("Bajaj Enterprises") == "bajaj"
    assert normalize_business_name("Adani Trading Company") == "adani"
    assert normalize_business_name("Sharma Trading Co.") == "sharma"

    # France
    assert normalize_business_name("Carrefour Supermarché SAS") == "carrefour supermarche"
    assert normalize_business_name("Total Société Anonyme") == "total"
    assert normalize_business_name("Renault SA") == "renault"
    assert normalize_business_name("Orange SARL") == "orange"
    assert normalize_business_name("Boulangerie EURL") == "boulangerie"
    assert normalize_business_name("Dupond SNC") == "dupond"

    # Europe
    assert normalize_business_name("Siemens AG") == "siemens"
    assert normalize_business_name("Bosch GmbH") == "bosch"
    assert normalize_business_name("Heineken NV") == "heineken"
    assert normalize_business_name("Philips BV") == "philips"
    assert normalize_business_name("Ferrari SpA") == "ferrari"


def test_normalize_business_name_edge_cases():
    """Verify name normalization boundary handling and special characters."""
    # Special character expansion
    assert normalize_business_name("Barnes & Noble") == "barnes and noble"
    assert normalize_business_name("User@Domain") == "user at domain"

    # String with only legal suffix
    assert normalize_business_name("LLC") == ""
    assert normalize_business_name("Pvt Ltd") == ""

    # Digits and mixed names
    assert normalize_business_name("3M Innovation Inc") == "3m innovation"
    assert normalize_business_name("7-Eleven Corp") == "7 eleven"

    # None and empty
    assert normalize_business_name(None) == ""
    assert normalize_business_name("") == ""
    assert normalize_business_name("   ") == ""
    assert normalize_business_name("nan") == ""


def test_normalize_business_address_expansions():
    """Verify street, unit, and state abbreviation expansions (D-11, PREP-03)."""
    # US address
    addr_us = "123 Main St, Ste 400, NY"
    norm_us = normalize_business_address(addr_us)
    assert "street" in norm_us
    assert "suite" in norm_us
    assert "new york" in norm_us
    assert norm_us == "123 main street suite 400 new york"

    # Road types
    assert "road" in normalize_business_address("45 Park Rd")
    assert "avenue" in normalize_business_address("10 5th Ave")
    assert "boulevard" in normalize_business_address("200 Sunset Blvd")
    assert "drive" in normalize_business_address("12 Ocean Dr")
    assert "highway" in normalize_business_address("Route 66 Hwy")
    assert "lane" in normalize_business_address("3 Penny Ln")
    assert "court" in normalize_business_address("7 King Ct")
    assert "parkway" in normalize_business_address("8 Garden Pkwy")

    # Indian states
    assert "maharashtra" in normalize_business_address("Plot 45, MG Rd, MH")
    assert "karnataka" in normalize_business_address("Electronic City, KA")
    assert "uttar pradesh" in normalize_business_address("Sector 62, Noida, UP")
    assert "tamil nadu" in normalize_business_address("Anna Salai, Chennai, TN")
    assert "delhi" in normalize_business_address("Connaught Place, DL")

    # Missing / Null
    assert normalize_business_address(None) == ""
    assert normalize_business_address("nan") == ""
    assert normalize_business_address("null") == ""


def test_create_search_text():
    """Verify name-boosted composite search text creation (D-12)."""
    text = create_search_text("Acme Corp", "100 Broadway St")
    assert text == "acme acme 100 broadway street"

    # Empty address
    assert create_search_text("Reliance Industries Pvt Ltd", None) == "reliance reliance"

    # Empty name
    assert create_search_text(None, "100 Broadway St") == "100 broadway street"

    # Both empty
    assert create_search_text(None, None) == ""
