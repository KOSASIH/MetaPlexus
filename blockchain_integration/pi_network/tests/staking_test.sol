pragma solidity ^0.8.0;

import "truffle/Assert.sol";
import "../contracts/Staking.sol";

contract StakingTest {
    Staking staking;

    function beforeEach() public {
        staking = new Staking();
    }

    function testStake() public {
        address staker = address(0x1);
        uint256 amount = 100;

        staking.stake(amount);

        Assert.equal(staking.stakers(staker), amount, "Stake amount is incorrect");
    }

    function testUnstake() public {
        address staker = address(0x1);
        uint256 amount = 100;

        staking.unstake(amount);

        Assert.equal(staking.stakers(staker), 0, "Stake amount is not zero");
    }
}
