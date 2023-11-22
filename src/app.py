from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import vectorizer

# Sample random text data
text_data = """
Once upon a time there were three bears, who lived together in a house of their own in a wood. One of them was a little, small wee bear; one was a middle-sized bear, and the other was a great, huge bear.
One day, after they had made porridge for their breakfast, they walked out into the wood while the porridge was cooling. And while they were walking, a little girl came into the house. This little girl had golden curls that tumbled down her back to her waist, and everyone called her by Goldilocks.
Goldilocks went inside. First she tasted the porridge of the great, huge bear, and that was far too hot for her. And then she tasted the porridge of the middle bear, and that was too cold for her. And then she went to the porridge of the little, small wee bear, and tasted that. And that was neither too hot nor too cold, but just right; and she liked it so well, that she ate it all up.
Then Goldilocks went upstairs into the bed chamber and first she lay down upon the bed of the great, huge bear, and then she lay down upon the bed of the middle bear and finally she lay down upon the bed of the little, small wee bear, and that was just right. So she covered herself up comfortably, and lay there until she fell fast asleep.
By this time, the three bears thought their porridge would be cool enough, so they came home to breakfast.
“SOMEBODY HAS BEEN AT MY PORRIDGE!” said the great huge bear, in his great huge voice.
“Somebody has been at my porridge!” said the middle bear, in his middle voice.
Then the little, small wee bear looked at his, and there was the spoon in the porridge pot, but the porridge was all gone.
“Somebody has been at my porridge, and has eaten it all up!” said the little, small wee bear, in his little, small wee voice.
Then the three bears went upstairs into their bedroom.
“SOMEBODY HAS BEEN LYING IN MY BED!” said the great, huge bear, in his great, rough, gruff voice.
“Somebody has been lying in my bed!” said the middle bear, in his middle voice.
And when the little, small, wee bear came to look at his bed, upon the pillow there was a pool of golden curls, and the angelic face of a little girl snoring away, fast asleep.
“Somebody has been lying in my bed, and here she is!” Said the little, small wee bear, in his little, small wee voice.
Goldilocks jumped off the bed and ran downstairs, out of the door and down the garden path. She ran and she ran until she reached the house of her grandmama. When she told her grandmama about the house of the three bears who lived in the wood, her granny said: “My my, what a wild imagination you have, child!”
"""

# Basic display of text
st.header("Semantic Wiki")
st.subheader("by Shaan Khosla, Shiqi Yang, and Teo Zeng")
st.write(
    "In this app, we will be analyzing the following text in the interactive area below. Feel free to replace it with your own text."
)

# Interactive text area
st.header("Interactive Text Area")
text_area = st.text_area("Edit the text", text_data, height=150)
st.set_option("deprecation.showPyplotGlobalUse", False)
# Generate and display a word cloud
st.header("Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
    text_area
)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
st.pyplot()

# Display a frequency distribution of the words
st.header("Frequency Distribution")
words = text_area.split()
word_counts = Counter(words)
df_words = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(
    by="Frequency", ascending=False
)
st.bar_chart(df_words.set_index("Word"))

# Run this script with:
# streamlit run your_script_name.py

# Vectorize the text
st.header("PCA")
st.write(
    "PCA (Principle Component Analysis) is a linear dimensionality reduction technique that transforms the data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. PCA can be used for data compression, noise reduction, and data visualization. It is particularly useful when you want to reduce the dimensionality of data with linear correlations. The main goal of PCA is to identify the most meaningful basis to re-express a dataset, seeking to highlight their similarities and differences. Since PCA is a linear method, it does not capture non-linear relationships well."
)
vectorizer = vectorizer.Vectorizer()
# splitting the text into sentences based on the period
sentences = text_area.split(".")
embeddings = vectorizer.vectorize(sentences)

# Initialize PCA with 3 components for 3D visualization
pca = PCA(n_components=3)

# Fit PCA on the embeddings and reduce the dimensionality
embeddings_pca_3d = pca.fit_transform(embeddings)

# Create a DataFrame for the PCA components
df_pca_3d = pd.DataFrame(embeddings_pca_3d, columns=["x", "y", "z"])
df_pca_3d["sentence"] = sentences

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    df_pca_3d,
    x="x",
    y="y",
    z="z",
    hover_data=["sentence"],
    title="3D PCA Visualization of Sentence Embeddings",
)

# Show the plot in Streamlit
st.plotly_chart(fig)

# t-SNE plot
st.header("t-SNE Plot")
st.write(
    "The t-SNE plot is displayed below. t-Distributed Stochastic Neighbor Embedding (t-SNE) is a machine learning algorithm for visualization developed by Laurens van der Maaten and Geoffrey Hinton. It is a nonlinear dimensionality reduction technique that is particularly well suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot."
)
# Initialize t-SNE with 3 components for 3D visualization
tsne = TSNE(n_components=3, perplexity=20, n_iter=300)

# Fit t-SNE on the embeddings and reduce the dimensionality
embeddings_tsne_3d = tsne.fit_transform(embeddings)

# Create a DataFrame for the t-SNE components
df_tsne_3d = pd.DataFrame(embeddings_tsne_3d, columns=["x", "y", "z"])
df_tsne_3d["sentence"] = sentences

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    df_tsne_3d,
    x="x",
    y="y",
    z="z",
    hover_data=["sentence"],
    title="3D t-SNE Visualization of Sentence Embeddings",
)

# Show the plot in Streamlit
st.plotly_chart(fig)
