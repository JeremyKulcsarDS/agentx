import fastapi
from fastapi import Header, HTTPException
import os
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
from geopy import distance

load_dotenv()

app = fastapi.FastAPI()

# Structure of the function inputs


class Coordinate(BaseModel):
    latitude: float
    longitude: float


class CoordinatePair(BaseModel):
    coordinate_0: Coordinate
    coordinate_1: Coordinate


@app.post('/geodesic')
async def geodesic(coordinate_pair: CoordinatePair, webhook_secret: Annotated[str, fastapi.Header()]):

    # make sure the request is sent from XEntropy
    if webhook_secret != os.environ.get('WEBHOOK_SECRET'):
        raise HTTPException(502, 'Webhook-Secret header cannot be none.')

    geodesic_distance = distance.distance(
        tuple(coordinate_pair.coordinate_0.dict().values()),
        tuple(coordinate_pair.coordinate_1.dict().values()),
    ).km

    result = {'geodesic_distance': geodesic_distance, 'unit': 'km'}

    return result
