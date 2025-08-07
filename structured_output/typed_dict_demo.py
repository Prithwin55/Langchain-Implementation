from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

person=Person(name="raj",age=34)
print(person)