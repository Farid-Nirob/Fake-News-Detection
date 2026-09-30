# Fake News Detection Using Machine Learning

This project applies machine learning techniques to classify news
articles as **fake** or **real**. Text preprocessing and TF-IDF feature
extraction are used to convert news articles into numerical features for
classification.

## Repository Structure

Fake-News-Detection/
├── README.md
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── docs/
│   └── Fake News Detection.pdf
└── .gitignore


## Dataset

The project uses two CSV files:

  Dataset      Class           Articles
  ------------ ----------- ------------
  Fake.csv   Fake news         23,481
  True.csv  Real news         21,417
  **Total**                  **44,898**

Both files contain the following columns:

-   title
-   text
-   subject
-   date

The notebook currently loads the dataset from Google Drive. The CSV
files are not included in this repository.

### Dataset Source

The dataset used in this project is available on Kaggle:

Fake and Real News Dataset — Clément Bisaillon  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Methodology

The main steps of the project are:

1.  Load the fake and real news datasets.
2.  Assign class labels to fake and real articles.
3.  Merge and shuffle the datasets.
4.  Clean and preprocess the news text.
5.  Split the data into **80% training** and **20% testing** sets.
6.  Convert the text into numerical features using **TF-IDF
    Vectorization**.
7.  Train and evaluate multiple machine learning classifiers.

## Models

The following classification algorithms were evaluated:

-   Logistic Regression
-   Decision Tree
-   Multinomial Naive Bayes
-   K-Nearest Neighbours (KNN)

## Results

The current reported test results are:

  Model                         Accuracy
  ------------------------- ------------
  Decision Tree               **99.64%**
  Logistic Regression         **98.78%**
  Multinomial Naive Bayes     **93.65%**
  K-Nearest Neighbours        **70.18%**

Decision Tree achieved the highest reported accuracy, followed by
Logistic Regression. Multinomial Naive Bayes also produced a relatively
high classification accuracy, while KNN showed considerably lower
performance on the TF-IDF feature representation.

These values represent the current experimental results reported in the
project. They can be revalidated when the complete notebook is rerun.

## Evaluation

Model performance was evaluated using confusion matrices and
classification metrics such as:

-   Accuracy
-   Precision
-   Recall
-   F1-score

The confusion matrices provide a detailed view of correctly and
incorrectly classified fake and real news articles.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   TF-IDF Vectorization
-   Google Colab / Jupyter Notebook

## How to Run

1.  Clone the repository.
2.  Open `notebooks/Fake_News_Detection.ipynb`.
3.  Obtain `Fake.csv` and `True.csv`.
4.  Update the dataset paths in the notebook according to their
    location.
5.  Install the required Python libraries.
6.  Run the notebook cells sequentially.

> The current notebook uses dataset paths from Google Drive. These paths
> need to be changed when running the project in another environment.

## Current Status

The initial machine learning pipeline and evaluation have been
completed. The notebook and project report are included in this
repository.

The project can be further improved by rerunning the complete pipeline
and organizing the newly generated outputs in the `results/` directory.

## Future Improvements

-   Make dataset loading independent of user-specific Google Drive
    paths.
-   Store evaluation figures and outputs in the `results/` directory.
-   Revalidate the reported model results.
-   Improve model comparison and experiment documentation.
-   Explore additional classification approaches.

