import pytest

from clif.terminology.hierarchy import EclError, EclUnsupported, HierarchyConstraint, parse_ecl


def test_ecl_operators(mapper):
    s = mapper.store
    assert parse_ecl("<< 64572001 |Disease|").matches("85189001", s)
    assert parse_ecl("< 74400008").matches("85189001", s)
    assert not parse_ecl("< 74400008").matches("74400008", s)
    assert parse_ecl("<< 74400008").matches("74400008", s)
    assert parse_ecl("<! 74400008").matches("85189001", s)
    assert parse_ecl(">> 85189001").matches("74400008", s)
    assert parse_ecl("> 85189001").matches("64572001", s)
    assert parse_ecl("74400008").matches("74400008", s)


def test_ecl_compound(mapper):
    s = mapper.store
    e = parse_ecl("<< 64572001 MINUS << 74400008")
    assert e.matches("22298006", s) and not e.matches("85189001", s)
    e = parse_ecl("(<< 71388002 OR << 123037004) AND << 138875005")
    assert e.matches("80146002", s) and e.matches("66754008", s) and not e.matches("74400008", s)


def test_ecl_unsupported_is_explicit():
    with pytest.raises(EclUnsupported):
        parse_ecl("<< 404684003 : 363698007 = << 39057004")
    with pytest.raises(EclError):
        parse_ecl("<< banana")


def test_named_hierarchies(mapper):
    c = HierarchyConstraint.build(["procedure"], None)
    assert c.allows("80146002", mapper.store)[0]
    assert not c.allows("74400008", mapper.store)[0]
    with pytest.raises(ValueError):
        HierarchyConstraint.build(["nonsense"], None)
