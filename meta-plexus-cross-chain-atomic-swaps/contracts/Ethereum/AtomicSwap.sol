pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract AtomicSwap {
    address public owner;
    mapping (address => mapping (address => uint256)) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function swap(address _tokenA, address _tokenB, uint256 _amountA, uint256 _amountB) public {
        // implement atomic swap logic here
    }

    function getBalance(address _token, address _user) public view returns (uint256) {
        return balances[_token][_user];
    }
}
