import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------------------------------------------------------------------------
# Name: StatsPeriod
# Type: Class
# Author: Hashim Chaudhry
# Description:  Computes the periodic statistics of a dataset over a particular time period based
#               on the date column that is passed to the function. This is a valid transformer
#               that returns a list of dataframes based on the attributes passed to it and can be
#               used in pipelines. The indices of the new dataframe will be the dates
# Returns:      Returns a list of dataframes that contain the period-based data. Only numerical data will
#               be saved, but category based organization will also contain an extra column for the category
#               under which the dataframe was organized
# Version:      v1.0.0
# Attributes:   Listed below
#               - (str) period:         The period in which to combine data over, defaults to monthly period
#                                       See pandas to_period function for valid identifiers on the period
#               - (bool) drop:          Whether to drop the original time column. Defaults to False
#               - (str) dates_col:      The column where the dates are located in the dataframe
#               - (str) category:       If by_category is true, then the transformer will separate the data
#                                       by the category column specified
#               - (bool) by_category:   Whether to split data into categories
#               - (string) aggregate:   How to aggregate data (currently supports mean and sum)
#               - (bool) drop_dups:     Whether to drop duplicate times
#               - (bool) return_df:     Whether to only return first category or not of filtered array
#               - (int)  return_index:  The index to return. Only necessary if return_df is True and by_category is True
# -------------------------------------------------------------------------------------------------

class StatsPeriod(BaseEstimator, TransformerMixin):
    def __init__(self, period="M", drop=False, dates_col="Dates", category="city_name", by_category=False,
                 aggregate="mean", drop_dups=None, df=False, df_index=0):
        self.period = period
        self.drop = drop
        self.dates_col = dates_col
        self.category = category
        self.by_category = by_category
        self.aggregate = aggregate
        self.drop_dups = drop_dups
        self.df = df
        self.df_index = df_index

    def fit(self, x, y=None):
        return self

    # Get categories through which we are ordering
    def get_categories(self, df):
        cats = df[self.category].value_counts().index.tolist()
        return cats

    # Drop unneeded columns
    def drop_cols(self, df):
        if self.drop_dups is not None:
            df = df.drop_duplicates(subset=self.drop_dups).copy()

        # Convert dates to datetime objects
        df["Date"] = pd.to_datetime(df[self.dates_col], errors='coerce', utc=True).dt.tz_localize(None)
        if self.drop:
            df.drop([self.dates_col], axis=1)
        return df

    # Combines a dataframes date indices through some period by some metric
    def combine(self, df):
        if self.aggregate == "mean":
            return df.groupby("Date").mean()  # append the dataframe to the datasets array
        elif self.aggregate == "sum":
            return df.groupby("Date").sum()
        return df.groupby("Date").sum()

    def transform(self, x):
        # If we are organizing by city, do so
        if self.by_category:
            cats = self.get_categories(x)
            datasets = []  # Dictionary to hold our data

            # Compute a dataframe for each city
            for item in cats:
                # Get the dataframe only associated with a particular city
                bool_array = x[self.category] == item
                df = x[bool_array].copy()  # Ensure to create a copy so that we aren't working with a view

                # Drop any categories based on attributes
                df = self.drop_cols(df)

                # Drop all categorical variables (type str)
                df_numeric = df.select_dtypes(["number", "datetime"]).copy()

                # Convert dataset to specific period and compute mean based on period
                df_numeric["Date"] = df_numeric["Date"].dt.to_period(self.period)
                df_numeric = self.combine(df_numeric)
                print(x[self.category][bool_array])
                df_numeric[self.category] = item
                datasets.append(df_numeric)

            if self.df:
                return datasets[self.df_index]
            else:
                return datasets

        else:
            # Drop any rows that contain any missing data
            datasets = []
            df = x.copy()
            # Drop any categories based on attributes
            df = self.drop_cols(df)

            # Drop all categorical variables (type str)
            df_numeric = df.select_dtypes(["number", "datetime"]).copy()

            # Convert dataset to specific period and compute mean based on period
            df_numeric["Date"] = df_numeric["Date"].dt.to_period(self.period)
            datasets.append(self.combine(df_numeric))

            if self.df:
                return datasets[0]
            else:
                return datasets


# -------------------------------------------------------------------------------------------------
# Name: DropCols
# Author: Hashim Chaudhry
# Description:  Drops specific columns in dataframe, valid transformer for pipelines in sklearn
# Returns:      Returns a dataframe
# Version: v1.0.0
# Attributes: Listed below
#               - (str) cols:       The columns to drop in the dataframe
# -------------------------------------------------------------------------------------------------
class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.drop(columns=self.cols)
