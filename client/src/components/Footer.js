import React from 'react';
import { useMetaPlexus } from '../hooks/useMetaPlexus';

const Footer = () => {
  const { metaPlexus } = useMetaPlexus();

  return (
    <footer>
      <p>&copy; 2024 MetaPlexus</p>
      <p>
        Powered by{' '}
        <a href="https://metaplexus.io" target="_blank" rel="noopener noreferrer">
          MetaPlexus
        </a>
      </p>
      <p>
        Contract address: {metaPlexus.address}
      </p>
    </footer>
  );
};

export default Footer;
