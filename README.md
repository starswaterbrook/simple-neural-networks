# Simple Neural Networks

This project aims to deliver a simple implementation of various neural network types using basic math tools. It is supposed to help with understanding the inner workings of these networks, often hidden behind abstraction in popular machine learning libraries. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Tests and Styling](#tests-and-styling)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/starswaterbrook/simple-neural-networks.git
    cd simple-neural-networks
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Usage is rather straightforward and showcased in [`mlp_train.py`](mlp_train.py) and [`mlp_load.py`](mlp_load.py)

## Tests and Styling

This projects uses [`tox`](https://github.com/tox-dev/tox), please check [`tox.ini`](tox.ini) for available commands.  

Example usage:
- Run tests
    ```
    tox -e run-tests
    ```
    or
    ```
    tox -e run-verbose-tests
    ```
    for additional output information.  

- Fix code style
    ```
    tox -e fix-styling
    ```

DO NOT change `requirements.txt` directly, use:
```
pip-compile requirements.in
```
when adding a new dependency. Then, to make sure it has been generated correctly, use:
```
./env_scripts/check_requirements.sh
```
When making a pull request, please try to adhere to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) specification.

## License
This project is under [MIT License](https://opensource.org/license/mit)