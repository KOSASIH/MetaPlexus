**blockchain.py**
```python
import hashlib
import hmac
import json
import time
from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigdecode_der
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf import hkdf
from web3 import Web3, HTTPProvider

# Load blockchain configuration
with open("blockchain_config.json", "r") as f:
    config = json.load(f)

# Define blockchain class
class Blockchain:
    def __init__(self, private_key_path, contract_address):
        self.private_key_path = private_key_path
        self.contract_address = contract_address
        self.web3 = Web3(HTTPProvider(config["rpc_url"]))
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=config["contract_abi"])

    def generate_token(self, user_id, amount):
        # Generate token using elliptic curve cryptography
        private_key = SigningKey.from_pem(open(self.private_key_path, "r").read())
        public_key = private_key.get_verifying_key()
        token_data = json.dumps({"user_id": user_id, "amount": amount})
        signature = private_key.sign(token_data.encode(), hashfunc=hashlib.sha256)
        token = {"token_data": token_data, "signature": signature.hex()}
        return token

    def verify_token(self, token):
        # Verify token using elliptic curve cryptography
        public_key = SigningKey.from_pem(open(self.private_key_path, "r").read()).get_verifying_key()
        token_data = token["token_data"]
        signature = bytes.fromhex(token["signature"])
        try:
            public_key.verify(signature, token_data.encode(), hashfunc=hashlib.sha256)
            return True
        except:
            return False

    def create_transaction(self, from_address, to_address, amount):
        # Create transaction using Web3
        tx = self.web3.eth.account.sign_transaction({
            "from": from_address,
            "to": to_address,
            "value": amount,
            "gas": 20000,
            "gasPrice": self.web3.eth.gas_price
        }, private_key=open(self.private_key_path, "r").read())
        return tx

    def execute_smart_contract(self, function_name, *args):
        # Execute smart contract function using Web3
        tx_hash = self.contract.functions[function_name](*args).transact()
        self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash

    def start_blockchain_system(self):
        # Initialize blockchain system
        while True:
            # Monitor for new transactions
            new_transactions = self.web3.eth.get_new_pending_transactions()

            # Process transactions
            for tx in new_transactions:
                # Verify token and execute smart contract function
                token = self.verify_token(tx["token"])
                if token:
                    self.execute_smart_contract("reward_user", tx["from"], tx["amount"])
                else:
                    print("Invalid token!")

            # Mine new block
            self.web3.eth.mine()

# Create blockchain instance
blockchain_system = Blockchain("private_key.pem", "0x...contract_address...")

# Start blockchain system
blockchain_system.start_blockchain_system()
