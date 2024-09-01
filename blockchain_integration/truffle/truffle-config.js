// truffle-config.js (Ethereum)
module.exports = {
  networks: {
    ethereum: {
      provider: () => new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'),
      gas: 5000000,
      gasPrice: 20e9,
    },
  },
};

// bnb-cli.config.js (Binance Smart Chain)
module.exports = {
  networks: {
    bsc: {
      provider: () => new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bnb/'),
      gas: 5000000,
      gasPrice: 20e9,
    },
  },
};

// polygon-cli.config.js (Polygon)
module.exports = {
  networks: {
    polygon: {
      provider: () => new Web3.providers.HttpProvider('https://polygon-rpc.com/'),
      gas: 5000000,
      gasPrice: 20e9,
    },
  },
};

// solana-cli.config.js (Solana)
module.exports = {
  networks: {
    solana: {
      provider: () => new SolanaProvider('https://api.mainnet-beta.solana.com'),
      gas: 5000000,
      gasPrice: 20e9,
    },
  },
};
