# Functions to ease data loading and exploration
import pandas as pd
import seaborn as sns


def create_pairplot(data, categories=None):
    if categories:
        sns.pairplot(data[categories])
    else:
        sns.pairplot(data)


def corr_matrix(data, category=None):
    matrix = data.corr()
    if category:
        return matrix["price actual"].sort_values(ascending=False)
    else:
        return matrix


def get_attribs(data, categories=None):
    if categories:
        return data[categories]
    else:
        return data
