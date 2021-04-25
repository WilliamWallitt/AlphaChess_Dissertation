# AlphaChess

# As this project requires several large datasets - https://drive.google.com/drive/folders/1WUoptZkBIn15QVEXwLL0Omyz6FmvVNHq contains the full project (its around 9gb due to the preprocessed PGN files)
# If you download the entire project you can run each stage of AlphaChess using the commands below:

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install chess
pip install tensorflow
pip install numpy
pip install tensorflow-probability
pip install matplotlib
pip install stockfish
```
## Usage - using google drive file

Navigate to the main.py file in the project's root directory

```python
    # training of the SL policy network
    AlphaChess().policy_network_supervised_learning(dataset_path="Data/human_chess_moves.pgn", preprocessing=False,
                                                  batch_size=64, filters=256, epochs=15)
    # training of the RL policy network, requires if __name__ == "__main__": guard, due to multiprocessing
    AlphaChess().policy_network_reinforcement_learning()
    # self play games of the RL policy network
    AlphaChess().policy_network_reinforcement_learning(generate_self_play_games=True, num_self_play_games=16)
    # training of the value network
    AlphaChess().value_network_supervised_learning(preprocessing=False, batch_size=64, filters=256, epochs=15,
                                                   raw_input=False)
    # Neural network MCTS vs standard MCTS
    AlphaChess().neural_network_mcts_vs_base_mcts(sl_policy_network=False, no_policy_network=False, no_value_network=False)
    # Neural network MCTS vs human player
    AlphaChess().neural_network_mcts_vs_human()
    # Neural network MCTS vs minimax search with value network
    AlphaChess().minimax_vs_base_mcts()

```






