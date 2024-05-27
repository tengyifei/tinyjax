# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tinygrad.nn.datasets import mnist
from tqdm import trange


class Model:
    def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 3),
            Tensor.relu,
            lambda x: Tensor.avg_pool2d(
                x,
                stride=(
                    2,
                    2,
                ),
            ),
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            lambda x: Tensor.avg_pool2d(
                x,
                stride=(
                    2,
                    2,
                ),
            ),
            lambda x: x.flatten(1),
            nn.Linear(1600, 256),
            Tensor.relu,
            nn.Linear(256, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()

    # TODO: remove this when HIP is fixed
    X_train, X_test = X_train.float(), X_test.float()

    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model))

    @TinyJit
    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(512, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = (
                model(X_train[samples])
                .sparse_categorical_crossentropy(Y_train[samples])
                .backward()
            )
            opt.step()
            return loss

    @TinyJit
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(70)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
