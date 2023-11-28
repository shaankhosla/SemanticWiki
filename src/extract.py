import wikipedia
import pandas as pd


def getwikidata(query):
    try:
        page = wikipedia.page(query)
    except ValueError:
        print(f"This query {query} not match any page.")
        return

    url = page.url
    q = "_".join(query.split(" "))

    data = {
        "title": [],
        "contents": [],
        "summaries": [],
        "urls": [],
        "categories": [],
        "links": [],
    }
    data["title"].append(q)
    data["summaries"].append(page.summary)
    data["categories"].append(",".join(page.categories))
    data["links"].append(",".join(page.links))
    data["urls"].append(page.url)
    if url.split("/")[-1] == q:
        ## query is the title of page in wiki
        data["contents"].append(page.content)

    else:
        data["contents"].append("")
        print(len(page.links))
        for link in page.links:
            print(link)
            try:
                new = wikipedia.page(link)
                data["title"].append(link)
                data["contents"].append(new.content)
                data["summaries"].append(new.summary)
                data["categories"].append(",".join(new.categories))
                data["links"].append(",".join(new.links))
                data["urls"].append(new.url)
            except ValueError:
                print(f"This id {link} not match any page")

    return pd.DataFrame(data)
