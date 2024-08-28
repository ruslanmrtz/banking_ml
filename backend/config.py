from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

URL = 'postgresql://club_owner:fwJIzm8od1pR@ep-gentle-base-a2dy8acg.eu-central-1.aws.neon.tech/club?options=-csearch_path%3Ddbo,cd'

engine = create_engine(URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

