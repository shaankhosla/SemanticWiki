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

    st.write(
        """
    A word cloud is a visual representation where words from a text are displayed in various sizes based on their frequency or importance.
    Words appearing more frequently are shown larger, making it easy to identify prominent themes or topics at a glance.
    """
    )

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="black",
        contour_width=3,
        contour_color="steelblue",
    ).generate(" ".join(wiki_data_df["content"].values.tolist()))

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()


def frequency_plot():
    st.header("Frequency Distribution")
    text = " ".join(wiki_data_df["content"].values.tolist())
    word_counts = Counter(text.split())
    df_words = pd.DataFrame(
        word_counts.items(),
        columns=["Word", "Frequency"],
    ).sort_values(by="Frequency", ascending=False)

    # Using Plotly for more control over the chart
    fig = px.bar(df_words, x="Word", y="Frequency")
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig)

    # Analyze and write a summary paragraph
    total_words = sum(word_counts.values())
    most_common_word, highest_frequency = df_words.iloc[0]
    unique_words = len(word_counts)
    words_appearing_once = sum(1 for word, count in word_counts.items() if count == 1)

    summary = f"""
    The chart above displays the frequency distribution of words in the dataset.
    A total of {total_words} words were analyzed, among which there are {unique_words} unique words.
    The most common word is '{most_common_word}' with a frequency of {highest_frequency}.
    Interestingly, {words_appearing_once} words appear only once in the entire dataset.
    """
    st.write(summary)


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
        wiki_data_df = getwikidata(10)
        print(wiki_data_df)
        st.session_state.wiki_data_df = wiki_data_df
    main()
