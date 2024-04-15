import motor.motor_asyncio
from customer import Customer

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017/')
db = client.DemoDB
colCustomer = db.Customer

async def create(c):
    await colCustomer.insert_one(c)
    return c

async def read_one(id):
    return await colCustomer.find_one({'CustomerId': id})

async def read_all():
    data = []
    cs = colCustomer.find({})
    
    async for document in cs:
        data.append(Customer(**document))
    
    return data

async def update(id, name):
    await colCustomer.update_one({'CustomerId': id}, {'$set': {'Name': name}})
    
    return await colCustomer.find_one({'CustomerId': id})

async def delete(id):
    return await colCustomer.delete_one({'CustomerId': id})
