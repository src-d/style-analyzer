# React visualizer for features

## Installation

Install node.js and npm and run

    npm install

in the visualizer directory to install dependencies.

Train a model and put it in the `models` directory. Name it `model.asdf`.

And install python dependencies

    pip install -r requirements.txt

## Build

To build the React client, run

    npm run build

in the visualizer directory.

## Usage

1. Run

        FLASK_APP=server.py python3 -m flask run --port 5001

    in the visualizer directory to run the python server.

2. Run

        npm run serve

    in the visualizer directory to launch the React client.


3. Go to the URL printed by the React client.
