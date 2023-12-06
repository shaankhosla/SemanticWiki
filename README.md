# SemanticWiki: Visualizing Wikipedia Data

SemanticWiki is an application designed to analyze and visualize Wikipedia data, providing insights into the structure and content of Wikipedia articles. It was developed by `Shaan Khosla`, `Shiqi Yang`, and `Teo Zeng`. This application uses Python and several powerful libraries to extract, process, and visualize data from Wikipedia.

## Data Extraction

In our data extraction process, we utilized the `wikipedia` Python package to gather information from Wikipedia. We began by fetching random page titles, dividing our requests into smaller chunks to adhere to Wikipedia's rate limits. Each title was then used to extract page details, including content and summaries, which included a retry mechanism for reliability.

To speed up the process, we employed Python's multiprocessing module, enabling parallel data collection. The resulting information from each page was structured into a schema and compiled into a Pandas DataFrame, which was later imported on the streamlit application.

## Features

  - **Word Cloud**: Creates a visually appealing word cloud to highlight the most frequent words in the Wikipedia dataset.
  - **Frequency Distribution**: Displays a bar chart of word frequencies, helping to identify the most common words and their prevalence.
  - **PCA Visualization**: Employs Principle Component Analysis (PCA) for a 3D scatter plot visualization, revealing patterns and relationships in the dataset.

## How to run

1. **Clone the Repository**: Clone this repository to your local machine using Git.

```
git clone https://ShaanKhosla/SemanticWiki.git
```

2. **Run the DashBoard**: Make sure you have Docker installed on your machine. Then, navigate to the root directory of the repository and run the following command:

```
docker compose up --build semanticwiki
```

## Testing the Application

The application is tested using the `pytest` framework, and the testing files are located in the `tests` directory. To run the tests, navigate to the root directory of the repository and run the following command:

```
docker compose up --build tests
```


## Technologies and packages

- **Python**: The primary programming language for data extraction/processing and visualization.
- **Streamlit**: For creating the interactive web application.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Plotly**: For creating a variety of data visualizations.
- **WordCloud**: To generate word cloud images.
- **Custom Modules (`extract`, `vectorizer`)**: For extracting data from Wikipedia and vectorizing text data. Especially, we used a huggingface transformer model to vectorize the text data.
- **Docker**: For containerizing the application and testing environment.
- **Pytest**: For testing the application.


## Acknowledgments

We would like to Thank Professor `Jeremy Curuksu` and the teaching assistants for their guidance and support throughout the project.
