from pydantic import BaseModel
from enum import Enum


class City(str, Enum):
    Bangalore = 'Bangalore'
    Pune = 'Pune'
    NewDelhi = 'New Delhi'

class PaymentTier(int, Enum):
    top = 1
    medium = 2
    low = 3

class Education(str, Enum):
    Bachelors = 'Bachelors'
    Masters = 'Masters'
    PHD = 'PHD'

class Gender(str, Enum):
    Male = 'Male'
    Female = 'Female'

class YesOrNo(str, Enum):
    Yes = 'Yes'
    No = 'No'

class LeaveOrNot(str, Enum):
    Stay = 'Stay'
    Leave = 'Leave'

# this class checks the input data types
class InputData(BaseModel):
    education: Education
    joining_year: int
    city: City
    payment_tier: PaymentTier
    age: int
    gender: Gender
    ever_benched: YesOrNo
    experience_in_current_domain: int


class Prediction(BaseModel):
    leave_or_not: LeaveOrNot