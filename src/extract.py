import multiprocessing as mp

import pandas as pd  # type: ignore
import wikipedia  # type: ignore
from more_itertools import chunked  # type: ignore
from retry import retry  # type: ignore

from src import schemas

wikipedia.set_rate_limiting(rate_limit=True)


@retry(tries=3, delay=2, backoff=2)
def get_page_from_title(title: str) -> schemas.WikiDocument | None:
    try:
        page = wikipedia.page(title=title, preload=True)
        doc = schemas.WikiDocument(
            content=page.content,
            summary=page.summary,
            title=title,
            links=page.links,
        )
        return doc

    except Exception as e:
        print(e)
        return None


@retry(tries=3, delay=2, backoff=2)
def get_random_title(pages: int = 1) -> list[str]:
    return wikipedia.random(pages=pages)


def getwikidata(n: int = 100) -> pd.DataFrame:
    # wikipedia blocks more than 10
    chunks = list(chunked(range(n), 10))
    titles = []
    for chunk in chunks:
        titles += get_random_title(pages=len(chunk))

    with mp.Pool(5) as p:
        results = list(p.map(get_page_from_title, titles))

    results = [model.model_dump() for model in results if model]
    df = pd.DataFrame(results)
    return df
