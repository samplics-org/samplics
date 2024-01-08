import numpy as np
import polars as pl

from samplics.apis import fit
from samplics.types import AuxVars, DirectEst, FitMethod, FitStats


def test_fit():
    area = np.linspace(start=1, stop=50, num=50).tolist()
    ssize_arr = np.unique(area, return_counts=True)
    ssize = dict(zip(ssize_arr[0], ssize_arr[1]))
    y_stderr = np.random.uniform(low=1, high=10, size=50)

    x_df = pl.DataFrame(
        {
            "x1": np.random.choice(a=[1, 2, 7, 5], size=50).tolist(),
            "x2": (13 * np.random.normal(size=50)).tolist(),
        }
    )
    x = AuxVars(x=x_df, domain=area)

    # y = 150 * np.random.normal(size=250)
    y = (
        2 * np.random.choice(a=[1, 2, 7, 5], size=50)
        + 3 * (13 * np.random.normal(size=50))
        + np.random.uniform(low=1, high=5, size=50)
    )

    y_hat = DirectEst(est=y, stderr=y_stderr, domain=area, ssize=ssize)

    y_fit = fit(y=y_hat, x=x, method=FitMethod.reml)

    isinstance(y_fit, FitStats)


from typing import Protocol


class Animal(Protocol):
    def speak():
        ...


class Dog:
    def speak(self):
        print("I am a dog")


class Cat:
    def speak(self):
        print("I am a cat")


def use(animal: Animal):
    match animal:
        case Dog():
            animal.speak()
        case Cat():
            animal.speak()
        case _:
            print("No animal")


miaou = Cat()

breakpoint()
