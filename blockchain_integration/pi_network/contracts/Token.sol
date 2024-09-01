pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Token is SafeERC20 {
    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Mapping of token allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Token name
    string public name;

    // Token symbol
    string public symbol;

    // Token total supply
    uint256 public totalSupply;

    // Reentrancy protection
    bool private reentrancyLock;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Constructor
    constructor(string memory _name, string memory _symbol, uint256 _totalSupply) public {
        name = _name;
        symbol = _symbol;
        totalSupply = _totalSupply;
        balances[msg.sender] = totalSupply;
    }

    // Transfer function with reentrancy protection
    function transfer(address to, uint256 value) public {
        require(!reentrancyLock, "Reentrancy lock is enabled");
        reentrancyLock = true;
        _transfer(msg.sender, to, value);
        reentrancyLock = false;
    }

    // Transfer function with input validation
    function _transfer(address from, address to, uint256 value) internal {
        require(from != address(0), "From address is zero");
        require(to != address(0), "To address is zero");
        require(value > 0, "Value is zero");
        require(balances[from] >= value, "Insufficient balance");
        balances[from] -= value;
        balances[to] += value;
        emit Transfer(from, to, value);
    }

    // Approval function with input validation
    function approve(address spender, uint256 value) public {
        require(spender != address(0), "Spender address is zero");
        require(value > 0, "Value is zero");
        allowances[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
    }

    // Allowance function with input validation
    function allowance(address owner, address spender) public view returns (uint256) {
        require(owner != address(0), "Owner address is zero");
        require(spender != address(0), "Spender address is zero");
        return allowances[owner][spender];
    }
}
