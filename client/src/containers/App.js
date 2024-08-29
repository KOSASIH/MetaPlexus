import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Web3Provider } from '@ethersproject/providers';
import { useWeb3React } from '@web3-react/core';
import { useMetaPlexus } from '../hooks/useMetaPlexus';
import Header from '../components/Header';
import Footer from '../components/Footer';
import Home from '../pages/Home';
import Deposit from '../pages/Deposit';
import Withdraw from '../pages/Withdraw';

const App = () => {
  const { account, library } = useWeb3React();
  const { metaPlexus } = useMetaPlexus();
  const [network, setNetwork] = useState(null);
  const [chainId, setChainId] = useState(null);

  useEffect(() => {
    if (library) {
      library.getNetwork().then((network) => setNetwork(network));
      library.getChainId().then((chainId) => setChainId(chainId));
    }
  }, [library]);

  return (
    <Web3Provider library={library}>
      <BrowserRouter>
        <Header />
        <Switch>
          <Route exact path="/" component={Home} />
          <Route path="/deposit" component={Deposit} />
          <Route path="/withdraw" component={Withdraw} />
        </Switch>
        <Footer />
      </BrowserRouter>
    </Web3Provider>
  );
};

export default App;
