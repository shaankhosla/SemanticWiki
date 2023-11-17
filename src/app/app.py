import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Sample random text data
text_data = """
Once upon a time in a land far, far away, there lived a brave knight who went on an adventure to seek a legendary treasure.
This treasure was said to be the most valuable in all the kingdoms. Along the way, the knight faced many challenges and met new friends.
"""

# Basic display of text
st.header("Basic Text Display")
st.write(text_data)

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
