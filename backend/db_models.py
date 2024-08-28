from sqlalchemy import Column, Integer
from config import Base


class ClientDB(Base):
    __tablename__ = 'Clients'

    AGREEMENT_RK = Column(Integer, primary_key=True)
    TARGET = Column(Integer)
    AGE = Column(Integer)
    SOCSTATUS_WORK_FL = Column(Integer)
    SOCSTATUS_PENS_FL = Column(Integer)
    GENDER = Column(Integer)
    CHILD_TOTAL = Column(Integer)
    DEPENDANTS = Column(Integer)
    PERSONAL_INCOME = Column(Integer)
    LOAN_NUM_TOTAL = Column(Integer)
    LOAN_NUM_CLOSED = Column(Integer)



