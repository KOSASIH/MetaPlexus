pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/ERC20.sol";

contract Token is ERC20 {
    // Name of the token
    string public name = "MetaPlexus Token";

    // Symbol of the token
    string public symbol = "MPT";

    // Decimals of the token
    uint8 public decimals = 18;

    // Total supply of the token
    uint256 public totalSupply = 100000000 * (10 ** decimals);

    // Mapping of user addresses to their balances
    mapping (address => uint256) public balances;

    // Event emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when tokens are approved for spending
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Function to transfer tokens
    function transfer(address to, uint256 value) public {
        // Check if the user has sufficient balance
        require(balances[msg.sender] >= value, "Insufficient balance");

        // Update the user's balance
        balances[msg.sender] = balances[msg.sender].sub(value);

        // Update the recipient's balance
        balances[to] = balances[to].add(value);

        // Emit the Transfer event
        emit Transfer(msg.sender, to, value);
    }

    // Function to approve tokens for spending
    function approve(address spender, uint256 value) public {
        // Update the allowance
        allowances[msg.sender][spender] = value;

        // Emit the Approval event
        emit Approval(msg.sender, spender, value);
    }
}
