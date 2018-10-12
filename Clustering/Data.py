import pandas as pd
import os
from Clustering.Preprocessing import *
import pprint

def load_data():
    path = "C:/Users/madfa/Downloads/Git Repository/Document Dictionary/Documents.csv"
    data = pd.read_csv(path,encoding="utf-8")
    return data

class ImportData:
  news_df = load_data()
  eda = exploratory_data_analysis(news_df['Content'][1])
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(eda)





