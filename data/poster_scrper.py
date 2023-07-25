import pandas as pd
import numpy as np
from PyMovieDb import IMDB
import json
from tqdm import tqdm
import time

def add_tt(x):
    x=str(x)
    reminders=7-len(x)
    tconst= f"tt{reminders*'0'}{x}"
    imdb = IMDB()
    res = imdb.get_by_id(tconst)
    if res =='{"status": 404, "message": "No Result Found!", "result_count": 0, "results": []}': 
          print("error")
          return 'https://icon-library.com/images/404-error-icon/404-error-icon-24.jpg'
    else:
          data= json.loads(res)
          print(data["poster"])
          return data["poster"]
     

df_link=pd.read_csv('links.csv')
df_link1=df_link.iloc[:2000]
df_link1["poster"]=df_link1["imdbId"].apply(add_tt)
df_link1.to_csv('df_link1.csv')
print("1 done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

time.sleep(30)
df_link2=df_link.iloc[2000:4000]
df_link2["poster"]=df_link2["imdbId"].apply(add_tt)
df_link2.to_csv('df_link2.csv')
time.sleep(10)
print("2 done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

df_link3=df_link.iloc[4000:6000]
df_link3["poster"]=df_link3["imdbId"].apply(add_tt)
df_link3.to_csv('df_link3.csv')
time.sleep(10)
print("3 done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

df_link4=df_link.iloc[6000:8000]
df_link4["poster"]=df_link4["imdbId"].apply(add_tt)
df_link4.to_csv('df_link4.csv')
time.sleep(10)
print("4 done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


df_link5=df_link.iloc[8000:]
df_link5["poster"]=df_link5["imdbId"].apply(add_tt)
df_link5.to_csv('df_link5.csv')
time.sleep(10)
print("5 done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")