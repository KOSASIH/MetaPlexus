import React from 'react';
import { Link } from 'react-router-dom';
import { useWeb3React } from '@web3-react/core';
import { Web3Provider } from '@ethersproject/providers';
import { useMetaPlexus } from '../hooks/useMetaPlexus';

const Header = () => {
  const { account, library } = useWeb3React();
  const { metaPlexus } = useMetaPlexus();
  const [balance, setBalance] = React.useState(0);

  React.useEffect(() => {
    if (account && library) {
      metaPlexus.balances(account).then((balance) => setBalance(balance));
    }
  }, [account, library, metaPlexus]);

  return (
    <header>
      <nav>
        <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/deposit">Deposit</Link>
          </li>
          <li>
            <Link to="/withdraw">Withdraw</Link>
          </li>
        </ul>
      </nav>
      <div>
        <span>Account: {account}</span>
        <span>Balance: {balance} tokens</span>
      </div>
    </header>
  );
};

export default Header;
