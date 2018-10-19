# React visualizer for features

## Installation

There are three components to install. First step is to clone the repository

    git clone git@github.com:src-d/style-analyzer.git

### Install babelfish

Follow the instructions of the [official
documentation](https://doc.bblf.sh/using-babelfish/getting-started.html) up until the drivers
installation. Instead of installing the recommended drivers, run the following command to install
the javascript driver version 1.2.0

    docker exec -it bblfshd bblfshctl driver install javascript bblfsh/javascript-driver:v1.2.0

### Install style-analyzer

In the `style-analyzer` directory, install style-analyzer with the `web` extras

    pip install -e '.[web]'

### Install the web client

1. Install node.js and npm

2. In the `visualizer` directory, install the web client by running

        npm install

## Usage

1. Train a model and put it in the `visualizer/models` directory. Name it `model.asdf`.

2. In the `visualizer` directory, launch the python server by running

        FLASK_APP=server.py python3 -m flask run --port 5001

3. You can launch the client in development mode (autoreload, better profiling) or production mode (faster):

    - (Development) In the `visualizer` directory, launch the development web client by running

            npm start

    - (Production) In the `visualizer` directory, launch the production web client by running

            npm run build && npm run serve

3. Go to [http://localhost:3000/](http://localhost:3000/).
