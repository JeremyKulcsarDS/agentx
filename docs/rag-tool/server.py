import fastapi
from fastapi import Header, HTTPException
import requests
import json
import os
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
import chromadb
import pandas

load_dotenv()

app = fastapi.FastAPI()

dataframe = pandas.read_parquet('etf.parquet')

dataframe = pandas.read_parquet('etf.parquet')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection",metadata={"hnsw:space": "cosine"})
collection.add(
    embeddings=[x.tolist() for x in dataframe.embedding.tolist()],
    documents=dataframe['Full Text'].tolist(),
    metadatas=[row.to_dict() for index, row in dataframe[['Symbol', 'ETF Name', 'Industry']].iterrows()],
    ids=[str(i) for i in range(len(dataframe))]
)

azure_base = os.environ.get('AZURE_OPENAI_API_BASE')
azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY')


class ETFDescription(BaseModel):
    etf_description: str

@app.post('/retrieve')
async def retrieve(etf_description:ETFDescription, webhook_secret:Annotated[str, fastapi.Header()]):
    
    # make sure the request is sent from XEntropy
    if webhook_secret != os.environ.get('WEBHOOK_SECRET'):
        raise HTTPException(502, 'Webhook-Secret header cannot be none.')

    response = requests.post(
        f'{azure_base}/openai/deployments/ada2/embeddings?api-version=2023-07-01-preview',
        json={
            "input": etf_description.etf_description
        },
        headers={
            "Content-Type": "application/json",
            "api-key": azure_api_key
        }
    ).json()

    embedding = response.get('data')[0].get('embedding')

    results = collection.query(
        query_embeddings=embedding,
        n_results=3
    )

    number_of_results = len(results.get('ids')[0])
    
    transposed_results = [
        {
            'id':results.get('ids')[0][i],
            'distances':results.get('distances')[0][i],
            'metadatas':results.get('metadatas')[0][i],
            'documents':results.get('documents')[0][i]
        }
        for i in range(number_of_results)
    ]
    
    return transposed_results
