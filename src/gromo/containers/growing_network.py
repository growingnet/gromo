import torch

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import (
    LinearAdditionGrowingModule,
    LinearGrowingModule,
)
from gromo.utils.utils import global_device


class GrowingNetwork(GrowingContainer):
    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 1,
        use_bias: bool = True,
        hidden_features: int = 10,
        device: torch.device = None,
    ):
        super(GrowingNetwork, self).__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        self.start_module = LinearAdditionGrowingModule(
            in_features=self.in_features, name="start"
        )
        self.l1 = LinearGrowingModule(
            in_features=self.in_features,
            out_features=hidden_features,
            use_bias=use_bias,
            post_layer_function=torch.nn.ReLU(),
            name="l1",
        )
        self.l2 = LinearGrowingModule(
            in_features=hidden_features,
            out_features=self.in_features,
            name="l2",
            use_bias=use_bias,
        )
        self.res_module = LinearAdditionGrowingModule(
            in_features=self.in_features,
            post_addition_function=torch.nn.ReLU(),
            name="res",
        )
        self.l3 = LinearGrowingModule(
            in_features=self.in_features,
            out_features=self.out_features,
            name="l3",
            use_bias=use_bias,
        )
        self.l4 = LinearGrowingModule(
            in_features=self.in_features,
            out_features=hidden_features,
            post_layer_function=torch.nn.ReLU(),
            name="l4",
            use_bias=use_bias,
        )
        self.l5 = LinearGrowingModule(
            in_features=hidden_features,
            out_features=self.out_features,
            name="l5",
            use_bias=use_bias,
        )
        self.end_module = LinearAdditionGrowingModule(
            in_features=self.out_features, name="end"
        )

        self.start_module.set_next_modules([self.l1, self.res_module])
        self.l1.previous_module = self.start_module
        self.l1.next_module = self.l2
        self.l2.previous_module = self.l1
        self.l2.next_module = self.res_module
        self.res_module.set_previous_modules([self.start_module, self.l2])
        self.res_module.set_next_modules([self.l3, self.l4])
        self.l3.previous_module = self.res_module
        self.l3.next_module = self.end_module
        self.l4.previous_module = self.res_module
        self.l4.next_module = self.l5
        self.l5.previous_module = self.l4
        self.l5.next_module = self.end_module
        self.end_module.set_previous_modules([self.l3, self.l5])

    def __str__(self, verbose=0):
        if verbose == 0:
            return super(GrowingNetwork, self).__str__()
        else:
            txt = [f"{self.__class__.__name__}:"]
            for layer in self.children():
                txt.append(layer.__str__(verbose=verbose))
            return "\n".join(txt)

    def forward(self, x):
        x = self.start_module(x)
        x1 = self.l1(x)
        x1 = self.l2(x1)
        x = self.res_module(x + x1)
        x1 = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return self.end_module(x + x1)

    def start_computing_s_m(self):
        for layer in self.children():
            layer.tensor_s.init()
            if isinstance(layer, LinearGrowingModule):
                layer.tensor_m.init()
                layer.store_input = True
                layer.store_pre_activity = True

    def update_s_m(self):
        for layer in self.children():
            if isinstance(layer, LinearGrowingModule):
                layer.tensor_m.update()
                layer.tensor_s.update()

    def pass_s_m(self, input_x, target_y, loss=torch.nn.MSELoss()):
        input_x = input_x.to(self.device)
        target_y = target_y.to(self.device)
        self.zero_grad()
        y = self(input_x)
        loss_value = loss(y, target_y)
        loss_value.backward()
        self.update_s_m()

    def stop_computing_s_m(self):
        for layer in self.children():
            layer.tensor_s.reset()
            if isinstance(layer, LinearGrowingModule):
                layer.tensor_m.reset()

            if isinstance(layer, LinearAdditionGrowingModule):
                if layer.previous_tensor_s is not None:
                    layer.previous_tensor_s.reset()
                if layer.previous_tensor_m is not None:
                    layer.previous_tensor_m.reset()
            layer.store_input = False
            layer.store_pre_activity = False


if __name__ == "__main__":
    from torchinfo import summary

    device = global_device()
    net = GrowingNetwork(5, 1, device=device)
    x_input = torch.randn(20, 5, device=device)
    y = net(x_input)
    torch.norm(y).backward()

    print(net)
    print(net.l1.layer.weight.device)

    summary(net, input_size=(1, 5), device=device)

    print(net.l1.layer.weight.device)

    for layer in net.children():
        print(layer.__str__(verbose=2))

    net.start_computing_s_m()

    print("=" * 80)
    for layer in net.children():
        print(layer.__str__(verbose=2))

    net.end_module.previous_tensor_s.init()
    net.end_module.previous_tensor_m.init()

    for _ in range(2):
        x_input = torch.randn(20, 5)
        # net.zero_grad()
        # y = net(x_input)
        # torch.norm(y).backward()
        net.pass_s_m(x_input, torch.zeros(20, 1))
        net.end_module.previous_tensor_s.update()
        net.end_module.previous_tensor_m.update()

    for layer in net.children():
        print(layer.__str__(verbose=2))

    for layer in net.children():
        if isinstance(layer, LinearGrowingModule):
            layer.compute_optimal_delta()

    for layer in net.children():
        print(layer.__str__(verbose=2))

    net.stop_computing_s_m()

    for layer in net.children():
        print(layer.__str__(verbose=2))
