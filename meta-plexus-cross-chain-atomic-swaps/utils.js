// Utility functions for MetaPlexus Cross-Chain Atomic Swaps

/**
 * Converts a BigNumber to a string
 * @param {BigNumber} value - The BigNumber to convert
 * @returns {string} - The string representation of the BigNumber
 */
function bigNumberToString(value) {
  return value.toString();
}

/**
 * Converts a string to a BigNumber
 * @param {string} value - The string to convert
 * @returns {BigNumber} - The BigNumber representation of the string
 */
function stringToBigNumber(value) {
  return new BigNumber(value);
}

export { bigNumberToString, stringToBigNumber };
