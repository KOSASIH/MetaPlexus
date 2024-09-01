import { createStore, combineReducers } from 'redux';
import atomicSwapReducer from './reducers/atomicSwapReducer';

const rootReducer = combineReducers({
    atomicSwap: atomicSwapReducer,
});

const store = createStore(rootReducer);

export default store;
