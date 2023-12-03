from pydantic import BaseModel


class WikiDocument(BaseModel):
    content: str
    title: str
    summary: str
    links: list[str]
