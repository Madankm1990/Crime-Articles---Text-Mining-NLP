import pandas as pd
import os


def load_data():
    path = "C:/Users/madfa/Downloads/Git Repository/Document Dictionary/Documents.csv"
    data = pd.read_csv(path,encoding="utf-8")
    return data

class ImportData:
  news_df = load_data()
  # news_df.head(10)
  print(news_df)





