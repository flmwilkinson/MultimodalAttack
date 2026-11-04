# -*- coding: utf-8 -*-
from .base import BaseStrategy, StrategyRegistry
from .pair import PairStrategy
from .crescendo import CrescendoStrategy
from .barrage import BarrageStrategy
from .mapa import MapaStrategy
from .tree import TreeStrategy
from .meta import MetaStrategy
from .bandit import BanditStrategy
from .evolve import EvolveStrategy
from .bon import BonStrategy
from .surrogate import SurrogateStrategy
from .bayesopt import BayesOptStrategy
from .optimize import OptimizeStrategy

StrategyRegistry.register("pair", PairStrategy)
StrategyRegistry.register("crescendo", CrescendoStrategy)
StrategyRegistry.register("barrage", BarrageStrategy)
StrategyRegistry.register("mapa", MapaStrategy)
StrategyRegistry.register("tree", TreeStrategy)
StrategyRegistry.register("meta", MetaStrategy)
StrategyRegistry.register("bandit", BanditStrategy)
StrategyRegistry.register("evolve", EvolveStrategy)
StrategyRegistry.register("bon", BonStrategy)
StrategyRegistry.register("surrogate", SurrogateStrategy)
StrategyRegistry.register("bayesopt", BayesOptStrategy)
StrategyRegistry.register("optimize", OptimizeStrategy)

__all__ = [
    "BaseStrategy", "StrategyRegistry",
    "PairStrategy", "CrescendoStrategy", "BarrageStrategy",
    "MapaStrategy", "TreeStrategy", "MetaStrategy",
    "BanditStrategy", "EvolveStrategy",
    "BonStrategy", "SurrogateStrategy", "BayesOptStrategy",
    "OptimizeStrategy",
]
