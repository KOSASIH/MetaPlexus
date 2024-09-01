const initialState = {
    tokenA: '',
    tokenB: '',
    amountA: 0,
    amountB: 0,
};

export default function atomicSwapReducer(state = initialState, action) {
    switch (action.type) {
        case 'INIT_ATOMIC_SWAP':
            return { ...state, tokenA: action.tokenA, tokenB: action.tokenB };
        case 'SWAP_TOKENS':
            return { ...state, amountA: action.amountA, amountB: action.amountB };
        default:
            return state;
    }
}
