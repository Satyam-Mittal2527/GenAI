from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Satyam"
    age : Optional[int]= None
    email: EmailStr
    cgpa: float= Field(gt=0, lt=10, default = 0, description = """A Decimal value 
    representing the value of the student""")

new_student = {
    'age': '20',
    'email': 'abc@gmail.com',
    'cgpa': 8
}
student = Student(**new_student)
print(student)
print(type(student))
