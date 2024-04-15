from pydantic import BaseModel

class Customer(BaseModel):
    Customer: str
    name: str