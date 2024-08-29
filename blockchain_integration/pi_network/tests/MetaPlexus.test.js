const { expect } = require('chai');
const { web3 } = require('@openzeppelin/test-helpers');
const { MetaPlexus } = require('../contracts/MetaPlexus.sol');
const { Token } = require('../contracts/Token.sol');

describe('MetaPlexus', () => {
  let metaPlexus;
  let token;
  let owner;
  let user;

  beforeEach(async () => {
    // Deploy the Token contract
    token = await Token.new({ from: owner });

    // Deploy the MetaPlexus contract
    metaPlexus = await MetaPlexus.new(token.address, { from: owner });

    // Set the token address in the MetaPlexus contract
    await metaPlexus.setTokenAddress(token.address, { from: owner });

    // Transfer ownership of the Token contract to the MetaPlexus contract
    await token.transferOwnership(metaPlexus.address, { from: owner });

    // Set the user account
    user = web3.eth.accounts[1];
  });

  describe('deposit', () => {
    it('should allow users to deposit tokens', async () => {
      // Mint tokens to the user
      await token.mint(user, 100, { from: owner });

      // Deposit tokens to the MetaPlexus contract
      await metaPlexus.deposit(50, { from: user });

      // Check the user's balance
      expect(await metaPlexus.balances(user)).to.be.equal(50);
    });

    it('should emit the Deposit event', async () => {
      // Mint tokens to the user
      await token.mint(user, 100, { from: owner });

      // Deposit tokens to the MetaPlexus contract
      const tx = await metaPlexus.deposit(50, { from: user });

      // Check the event emission
      expect(tx.logs[0].event).to.be.equal('Deposit');
      expect(tx.logs[0].args.user).to.be.equal(user);
      expect(tx.logs[0].args.amount).to.be.equal(50);
    });
  });

  describe('withdraw', () => {
    it('should allow users to withdraw tokens', async () => {
      // Deposit tokens to the MetaPlexus contract
      await metaPlexus.deposit(50, { from: user });

      // Withdraw tokens from the MetaPlexus contract
      await metaPlexus.withdraw(20, { from: user });

      // Check the user's balance
      expect(await metaPlexus.balances(user)).to.be.equal(30);
    });

    it('should emit the Withdrawal event', async () => {
      // Deposit tokens to the MetaPlexus contract
      await metaPlexus.deposit(50, { from: user });

      // Withdraw tokens from the MetaPlexus contract
      const tx = await metaPlexus.withdraw(20, { from: user });

      // Check the event emission
      expect(tx.logs[0].event).to.be.equal('Withdrawal');
      expect(tx.logs[0].args.user).to.be.equal(user);
      expect(tx.logs[0].args.amount).to.be.equal(20);
    });
  });
});
