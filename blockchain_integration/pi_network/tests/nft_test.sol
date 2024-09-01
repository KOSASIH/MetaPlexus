pragma solidity ^0.8.0;

import "truffle/Assert.sol";
import "../contracts/NFT.sol";

contract NFTTest {
    NFT nft;

    function beforeEach() public {
        nft = new NFT("MyNFT", "MNFT");
    }

    function testMint() public {
        address to = address(0x1);
        string memory metadata = "https://example.com/metadata";

        nft.mint(to, metadata);

        Assert.equal(nft.owners(1), to, "Owner is incorrect");
        Assert.equal(nft.metadata(1), metadata, "Metadata is incorrect");
    }

    function testBurn() public {
        address owner = address(0x1);
        uint256 tokenId = 1;

        nft.burn(tokenId);

        Assert.equal(nft.owners(tokenId), address(0), "Owner is not zero");
        Assert.equal(nft.metadata(tokenId), "", "Metadata is not empty");
    }
}
