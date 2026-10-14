"""seru_core: reference implementation of the staged-constraint-handling
dynamic reconfiguration method for automotive final-assembly seru systems.

Modules
-------
instances    synthetic instance generator (Section 5.1, Table 4)
constraints  C1/C4 structural checks and C2 sequencing rules (Section 3.3)
operators    merge / split / reassign + restore + crossover (Section 4.2)
des          SimPy-based DES evaluator with C2/C3 checking (Section 4.3)
surrogate    ridge surrogates with leave-one-out uncertainty (Section 4.4)
nsga3        surrogate-assisted NSGA-III with error control (Algorithm 2)
"""

from .instances import Instance, initial_config, make_instance
from .constraints import (Config, check_c1, check_c4, fixtures_anchored,
                          in_omega_st)
from .operators import (crossover, merge, random_operator, reassign, restore,
                        split, warm_start)
from .des import DESResult, evaluate
from .surrogate import RidgeSurrogate, features
from .nsga3 import SearchConfig, SearchResult, surrogate_assisted_nsga3

__version__ = "0.1.0"
