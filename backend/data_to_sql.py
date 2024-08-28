import pandas as pd
from sqlalchemy import create_engine

URL = 'postgresql://club_owner:fwJIzm8od1pR@ep-gentle-base-a2dy8acg.eu-central-1.aws.neon.tech/club?options=-csearch_path%3Ddbo,cd'

engine = create_engine(URL)

df = pd.read_csv('../data/data.csv')
df.to_sql('Clients', engine, if_exists='replace', index=False)

print("Данные успешно загружены в таблицу PostgreSQL")