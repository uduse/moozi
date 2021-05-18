def test_mcts():
    from mozi import mcts
    config = mcts.Config()
    mcts.run_mcts(config, root)