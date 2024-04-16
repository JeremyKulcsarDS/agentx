import requests
from typing import Literal, Optional, Union
from pydantic import EmailStr
from eth_account import Account
from eth_account.messages import encode_defunct
from time import time


class Client():
    url = 'https://api.xentropy.co'

    def __init__(
        self,
        api_key: Optional[str] = None
    ):
        self.user_id = api_key.split('--')[0]
        self.api_key = api_key


    def summary(self):
        response = requests.get(
            f'{self.url}/users/',
            headers={
                'Api-Key': self.api_key
            }
        ).json()

        return response


    def delete_tool(self, tool):
        response = requests.delete(
            f'{self.url}/tools/{tool}',
            headers={
                'Api-Key': self.api_key
            }
        )
        return response.json()


    def register_ethereum_address(self, account: Account):
        timestamp = int(time())
        message = f'{self.user_id}-{timestamp}'
        message_hash = encode_defunct(text=message)
        signature = account.sign_message(message_hash).signature.hex()

        response = requests.post(
            f'{self.url}/users/register_ethereum_address',
            json={
                'user_id': self.user_id,
                'address': account.address,
                'timestamp': timestamp,
                'signature': signature
            },
            headers={
                'Api-Key': self.api_key
            }
        )
        return response


    def stable_coin_payout(self, amount: float, stable_coin: Literal['usdt', 'usdc', 'dai'], address: Optional[str] = None):
        response = requests.post(
            f'{self.url}/payout/stable_coin',
            json={
                'amount': amount,
                'address': address,
                'stable_coin': stable_coin,
            },
            headers={
                'Api-Key': self.api_key
            }
        ).json()

        return response


    def transfer_payout_to_balance(self, amount: float):
        response = requests.post(
            f'{self.url}/payout/balance',
            params={
                'amount': amount,
            },
            headers={
                'Api-Key': self.api_key
            }
        ).json()

        return response


    def log(
        self,
        parent_id: Optional[str] = None,
        tool: Optional[str] = None,
        start_after: Optional[float] = None,
        limit: int = 8,
    ):
        headers = {
            'Api-Key': self.api_key
        }

        if parent_id != None:
            response = requests.get(
                f'{self.endpoint}/log/{parent_id}',
                headers=headers,
            )
            return response.json()

        else:
            response = requests.get(
                f'{self.endpoint}/log',
                headers=headers,
                params={
                    'tool_id': tool,
                    'start_after': start_after,
                    'limit': limit,
                }
            )
            return response.json()
