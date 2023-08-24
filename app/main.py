from fastapi import FastAPI
from models.item import Item
from services.ticket_group import TicketGroupServices

import nltk
nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI( 
    title="IBB ML API",
    description="Indobara Machine Learning API",
    version="0.1.0" )

model = TicketGroupServices()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ticket-group/")
def predict_ticket(title: str):    
    model.text = title
    predict = model.predict_once()
    result = {
        "success" : True,
        "message" : "predict success",
        "data" : predict
    }
    
    return result


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}