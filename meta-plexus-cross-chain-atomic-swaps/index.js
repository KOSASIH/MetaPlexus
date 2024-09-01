// Import dependencies
import Web3 from 'web3';
import Wallet from './Wallet';
import BEP20Token from './BEP20Token';

// Set up Web3 provider
const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

// Set up wallet and token instances
const wallet = new Wallet('YOUR_PRIVATE_KEY', 'YOUR_PUBLIC_KEY', 'YOUR_ADDRESS');
const token = new BEP20Token('TOKEN_NAME', 'TOKEN_SYMBOL', 18, 'TOKEN_TOTAL_SUPPLY');

// Set up event listeners for token events
token.on('Transfer', (from, to, amount) => {
  console.log(`Transfer event: ${from} -> ${to} (${amount} tokens)`);
});

token.on('Approval', (owner, spender, amount) => {
  console.log(`Approval event: ${owner} approved ${spender} to spend ${amount} tokens`);
});

// Set up UI elements
const balanceElement = document.getElementById('balance');
const sendButton = document.getElementById('send-button');
const approveButton = document.getElementById('approve-button');
const transferButton = document.getElementById('transfer-button');

// Update balance display
balanceElement.innerText = `Balance: ${wallet.getBalance(token.address)} tokens`;

// Set up button event listeners
sendButton.addEventListener('click', () => {
  const recipient = document.getElementById('recipient-input').value;
  const amount = document.getElementById('amount-input').value;
  wallet.sendTokens(token.address, recipient, amount);
});

approveButton.addEventListener('click', () => {
  const spender = document.getElementById('spender-input').value;
  const amount = document.getElementById('amount-input').value;
  wallet.approveTokens(token.address, spender, amount);
});

transferButton.addEventListener('click', () => {
  const sender = document.getElementById('sender-input').value;
  const recipient = document.getElementById('recipient-input').value;
  const amount = document.getElementById('amount-input').value;
  wallet.transferTokens(token.address, sender, recipient, amount);
});
