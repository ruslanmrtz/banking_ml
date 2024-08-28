from pydantic import BaseModel
from enum import Enum

class Client(BaseModel):
    __tablename__ = 'Clients'

    AGE: int
    SOCSTATUS_WORK_FL: int
    SOCSTATUS_PENS_FL: int
    GENDER: int
    CHILD_TOTAL: int
    DEPENDANTS: int
    PERSONAL_INCOME: int
    LOAN_NUM_TOTAL: int
    LOAN_NUM_CLOSED: int

    class Config:
        from_attributes = True


class Prediction(int, Enum):
    respond = 1
    no_respond = 0
