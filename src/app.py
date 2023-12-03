from collections import Counter

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st  # type: ignore  # type: ignore
from wordcloud import WordCloud  # type: ignore

from src.extract import getwikidata
from src.vectorizer import Vectorizer


def title():
    st.header("Semantic Wiki")
    st.subheader("by Shaan Khosla, Shiqi Yang, and Teo Zeng")
    st.write(
        f"To build this app, we gathered {len(wiki_data_df)} random Wikipedia articles."
    )
    links = [len(x) for x in wiki_data_df.links.values.tolist()]
    word_counts = [len(x.split()) for x in wiki_data_df.content.values.tolist()]
    st.write(
        f"The articles had an average word count of {sum(word_counts)/len(word_counts):.1f} words"
    )
    st.write(f"The articles had an average of {sum(links)/len(links):.1f} links")

    table_of_contents = pd.DataFrame()
    table_of_contents["Title"] = wiki_data_df.title.values.tolist()
    table_of_contents["Number Outgoing Links"] = links
    table_of_contents["Word Count"] = word_counts
    table_of_contents = table_of_contents.reset_index(drop=True)
    st.table(table_of_contents)


def wordcloud_plot():
    st.set_option("deprecation.showPyplotGlobalUse", False)
    # Generate and display a word cloud
    st.header("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(wiki_data_df["content"].values.tolist())
    )
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()


def frequency_plot():
    # Display a frequency distribution of the words
    st.header("Frequency Distribution")
    text = " ".join(wiki_data_df["content"].values.tolist())
    word_counts = Counter(text.split())
    df_words = pd.DataFrame(
        word_counts.items(),
        columns=["Word", "Frequency"],
    ).sort_values(by="Frequency", ascending=False)
    st.bar_chart(df_words.set_index("Word"))


def scatter_plot():
    # Vectorize the text
    st.header("PCA")
    st.write(
        "PCA (Principle Component Analysis) is a linear dimensionality reduction technique that transforms the data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. PCA can be used for data compression, noise reduction, and data visualization. It is particularly useful when you want to reduce the dimensionality of data with linear correlations. The main goal of PCA is to identify the most meaningful basis to re-express a dataset, seeking to highlight their similarities and differences. Since PCA is a linear method, it does not capture non-linear relationships well."
    )
    if st.session_state.get("vectors", False):
        # load from state
        vectors = st.session_state.vectors
        vectors_reduced = st.session_state.vectors_reduced
    else:
        vectorizer = Vectorizer()
        docs = wiki_data_df.content.values.tolist()
        vectors = vectorizer.vectorize(docs)
        vectors_reduced = vectorizer.dimensionality_reduction(
            vectors=vectors,
            components=3,
        )

        st.session_state.vectors = vectors
        st.session_state.vectors_reduced = vectors_reduced

    # Create a DataFrame for the PCA components
    df_pca_3d = pd.DataFrame(vectors_reduced, columns=["x", "y", "z"])
    df_pca_3d["documents"] = docs

    # Create a 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df_pca_3d,
        x="x",
        y="y",
        z="z",
        hover_data=["documents"],
        title="3D PCA Visualization of Wikipedia Embeddings",
    )
    # Show the plot in Streamlit
    st.plotly_chart(fig)


def main():
    title()
    # text_area = st.text_area("Edit the text", text_data, height=150)
    wordcloud_plot()
    frequency_plot()
    scatter_plot()


if __name__ == "__main__":
    if st.session_state.get("wiki_data_df", False):
        # load from state
        wiki_data_df = st.session_state.wiki_data_df
    else:
        # gather
        wiki_data_df = getwikidata(2)
        st.session_state.wiki_data_df = wiki_data_df
    main()
