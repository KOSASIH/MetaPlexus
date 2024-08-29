const { deployer, web3 } = require('@openzeppelin/truffle');
const { MetaPlexus } = require('../contracts/MetaPlexus.sol');
const { Token } = require('../contracts/Token.sol');

module.exports = async (deployer) => {
  // Deploy the Token contract
  await deployer.deploy(Token, {
    from: deployer.accounts[0],
    gas: 4000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  // Deploy the MetaPlexus contract
  await deployer.deploy(MetaPlexus, Token.address, {
    from: deployer.accounts[0],
    gas: 4000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  // Initialize the MetaPlexus contract
  const metaPlexus = await MetaPlexus.deployed();
  await metaPlexus.initialize({
    from: deployer.accounts[0],
    gas: 2000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  // Set the token address in the MetaPlexus contract
  await metaPlexus.setTokenAddress(Token.address, {
    from: deployer.accounts[0],
    gas: 2000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  // Transfer ownership of the Token contract to the MetaPlexus contract
  const token = await Token.deployed();
  await token.transferOwnership(metaPlexus.address, {
    from: deployer.accounts[0],
    gas: 2000000,
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });
};
