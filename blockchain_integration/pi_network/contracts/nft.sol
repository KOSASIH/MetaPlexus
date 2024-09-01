pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract NFT is SafeERC721 {
    // Mapping of NFT owners
    mapping (uint256 => address) public owners;

    // Mapping of NFT metadata
    mapping (uint256 => string) public metadata;

    // NFT name
    string public name;

    // NFT symbol
    string public symbol;

    // Reentrancy protection
    bool private reentrancyLock;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 tokenId);

    // Constructor
    constructor(string memory _name, string memory _symbol) public {
        name = _name;
        symbol = _symbol;
    }

    // Transfer function with reentrancy protection
    function transfer(address to, uint256 tokenId) public {
        require(!reentrancyLock, "Reentrancy lock is enabled");
        reentrancyLock = true;
        _transfer(msg.sender, to, tokenId);
        reentrancyLock = false;
    }

    // Transfer function with input validation
    function _transfer(address from, address to, uint256 tokenId) internal {
        require(from != address(0), "From address is zero");
        require(to != address(0), "To address is zero");
        require(tokenId > 0, "Token ID is zero");
        require(owners[tokenId] == from, "Invalid owner");
        owners[tokenId] = to;
        emit Transfer(from, to, tokenId);
    }

       // Mint function with input validation
    function mint(address to, string memory _metadata) public {
        require(to != address(0), "To address is zero");
        require(_metadata != "", "Metadata is empty");
        uint256 newTokenId = uint256(keccak256(abi.encodePacked(_metadata)));
        owners[newTokenId] = to;
        metadata[newTokenId] = _metadata;
        emit Transfer(address(0), to, newTokenId);
    }

    // Burn function with input validation
    function burn(uint256 tokenId) public {
        require(tokenId > 0, "Token ID is zero");
        require(owners[tokenId] == msg.sender, "Invalid owner");
        delete owners[tokenId];
        delete metadata[tokenId];
        emit Transfer(msg.sender, address(0), tokenId);
    }

    // Approval function with input validation
    function approve(address approved, uint256 tokenId) public {
        require(approved != address(0), "Approved address is zero");
        require(tokenId > 0, "Token ID is zero");
        require(owners[tokenId] == msg.sender, "Invalid owner");
        emit Approval(msg.sender, approved, tokenId);
    }

    // Get token metadata function
    function getTokenMetadata(uint256 tokenId) public view returns (string memory) {
        require(tokenId > 0, "Token ID is zero");
        return metadata[tokenId];
    }
}
