pragma solidity ^0.8.0;

contract Staking {
    // Mapping of stakers
    mapping (address => uint256) public stakers;

    // Mapping of rewards
    mapping (address => uint256) public rewards;

    // Reentrancy protection
    bool private reentrancyLock;

    // Events
    event Stake(address indexed staker, uint256 amount);
    event Unstake(address indexed staker, uint256 amount);
    event Reward(address indexed staker, uint256 amount);

    // Stake function with reentrancy protection
    function stake(uint256 amount) public {
        require(!reentrancyLock, "Reentrancy lock is enabled");
        reentrancyLock = true;
        _stake(msg.sender, amount);
        reentrancyLock = false;
    }

    // Stake function with input validation
    function _stake(address staker, uint256 amount) internal {
        require(staker != address(0), "Staker address is zero");
        require(amount > 0, "Amount is zero");
        stakers[staker] += amount;
        emit Stake(staker, amount);
    }

    // Unstake function with reentrancy protection
    function unstake(uint256 amount) public {
        require(!reentrancyLock, "Reentrancy lock is enabled");
        reentrancyLock = true;
        _unstake(msg.sender, amount);
        reentrancyLock = false;
    }

    // Unstake function with input validation
    function _unstake(address staker, uint256 amount) internal {
        require(staker != address(0), "Staker address is zero");
        require(amount > 0, "Amount is zero");
        require(stakers[staker] >= amount, "Insufficient stake");
        stakers[staker] -= amount;
        emit Unstake(staker, amount);
    }

    // Reward function with reentrancy protection
    function reward(address staker) public {
        require(!reentrancyLock, "Reentrancy lock is enabled");
        reentrancyLock = true;
        _reward(staker);
        reentrancyLock = false;
    }

    // Reward function with input validation
    function _reward(address staker) internal {
        require(staker != address(0), "Staker address is zero");
        uint256 rewardAmount = calculateReward(staker);
        rewards[staker] += rewardAmount;
        emit Reward(staker, rewardAmount);
    }

    // Calculate reward function
    function calculateReward(address staker) internal view returns (uint256) {
        // TO DO: implement reward calculation logic
        return 0;
    }
}
