import unittest

from gromo.graph_network.dag_growing_network import GraphGrowingNetwork


class TestGraphGrowingNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.net = GraphGrowingNetwork(in_features=20, out_features=10, with_logger=False)
        self.net.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes={"type": "L", "size": self.net.neurons}
        )

    def test_init_empty_graph(self) -> None:
        self.net.init_empty_graph()
        assert len(self.net.dag.nodes) == 2
        assert len(self.net.dag.edges) == 0
        assert "start" in self.net.dag.nodes
        assert "end" in self.net.dag.nodes
        assert self.net.dag.in_degree("start") == 0
        assert self.net.dag.out_degree("start") == 0
        assert self.net.dag.in_degree("end") == 0
        assert self.net.dag.out_degree("end") == 0
        assert self.net.dag.nodes["start"]["size"] == 20
        assert self.net.dag.nodes["end"]["size"] == 10
        assert self.net.dag.nodes["start"]["type"] == "L"
        assert self.net.dag.nodes["end"]["type"] == "L"
        # assert self.net.dag.nodes["end"]["use_batch_norm"] == False

    def test_growth_histroy_step(self) -> None:
        self.net.growth_history_step(
            neurons_added=[("start", "1"), ("1", "end")],
            # neurons_updated=[("start", "end")],
            # nodes_added=["1"]
        )

        for edge in self.net.dag.edges:
            assert str(edge) in self.net.growth_history[self.net.global_step]
        assert self.net.growth_history[self.net.global_step][str(("start", "1"))] == 2
        assert self.net.growth_history[self.net.global_step][str(("1", "end"))] == 2
        assert self.net.growth_history[self.net.global_step]["1"] == 0

        self.net.growth_history_step(nodes_added=["1", "2"])
        assert self.net.growth_history[self.net.global_step]["1"] == 2
        assert "2" not in self.net.growth_history[self.net.global_step]


if __name__ == "__main__":
    unittest.main()
