import pandas as pd  # type: ignore

from src import extract


def test_getwikidata():
    results = extract.getwikidata(2)
    assert isinstance(results, pd.DataFrame)


def test_get_page_from_title():
    result = extract.get_page_from_title("Test case")
    assert (
        "In software engineering, a test case is a specification of the inputs, execution conditions"
        in result.content
    )
    assert "http://www.stickyminds.com/s.asp?F=S15689_ART_2" in result.links

    result = extract.get_page_from_title("sdihfoisdhgoishgoishdjofjoighosjfoishf")
    assert result is None
