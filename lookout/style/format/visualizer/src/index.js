import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import './css/index.css';
import registerServiceWorker from './registerServiceWorker';

ReactDOM.render(<App endpoint="http://127.0.0.1:5001/" />,
                document.getElementById('root'));
registerServiceWorker();
