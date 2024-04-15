from fastapi import FastAPI, HTTPException
from customer import Customer
from database import (
    create,
    read_all,
    read_one,
    update,
    delete,
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    'http://localhost:8000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/api/customer/', response_model = Customer)
async def PostCustomer(c: Customer):
    res = await create(c.dict())
    if res:
        return res    
    raise HTTPException(400, 'Invalid Create')

@app.get('/api/customer')
async def GetCustomer():
    return await read_all()

@app.get('/api/customer/{id}', response_model = Customer)
async def GetCustomerById(id):
    res = await read_one(id)
    if res:
        return res
    raise HTTPException(404, 'Not Found')

@app.put('/api/customer/{id}/', response_model = Customer)
async def PutCustomer(id: str, name: str):
    res = await update(id, name)
    if res:
        return res
    raise HTTPException(404, 'Not Found')

@app.delete('/api/customer/{id}')
async def DeleteCustomer(id):
    res = await delete(id)
    if res:
        return 'Delete Complete'
    raise HTTPException(404, 'Not Found')