# +
import fastapi
from fastapi import HTTPException
import requests
import os
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = fastapi.FastAPI()


class Address(BaseModel):
    address: str


@app.post('/geocoding')
async def geocoding(address: Address, webhook_secret: Annotated[str, fastapi.Header()]):

    # make sure the request is sent from XEntropy
    if webhook_secret != os.environ.get('WEBHOOK_SECRET'):
        raise HTTPException(502, 'Webhook-Secret header cannot be none.')

    geocoding = requests.get(
        'https://maps.googleapis.com/maps/api/geocode/json',
        params={
            'address': address.address,
            # YOUR Google Cloud API Key
            'key': os.environ.get('GOOGLE_CLOUD_API_KEY')
        }
    ).json()
    location = geocoding.get('results')[0].get('geometry').get('location')
    result = {'latitude': location.get(
        'lat'), 'longitude': location.get('lng')}

    return result
