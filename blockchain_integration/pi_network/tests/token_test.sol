pragma solidity ^0.8.0;

import "truffle/Assert.sol";
import "../contracts/Token.sol";

contract TokenTest {
    Token token;

    function beforeEach() public {
        token = new Token("MyToken", "MTK", 1000000);
    }

    function testTransfer() public {
        address from = address(0x1);
        address to = address(0x2);
        uint256 value = 100;

        token.transfer(from, to, value);

        Assert.equal(token.balances(from), 900000, "From balance is incorrect");
        Assert.equal(token.balances(to), 100, "To balance is incorrect");
    }

    function testApproval() public {
        address owner = address(0x1);
        address spender = address(0x2);
        uint256 value = 100;

        token.approve(owner, spender, value);

        Assert.equal(token.allowances(owner, spender), value, "Approval amount is incorrect");
    }
}
