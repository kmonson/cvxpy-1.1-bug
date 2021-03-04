# Most of this code was borrowed from StorageVet and adapted for our purposes
# https://github.com/epri-dev/StorageVET/blob/master/Scenario.py

from __future__ import annotations

import datetime
import os
from pathlib import Path
from pprint import pformat
import time
from typing import Any, Dict

import cvxpy
import numpy as np
import pandas as pd


CVXPY_VERBOSE = bool(os.environ.get('CVXPY_VERBOSE'))
CVXPY_TEST_FILE = Path(os.environ.get('CVXPY_TEST_FILE', "dump.txt"))

CVXPY_TEST_FILE.unlink(missing_ok=True)

CVXPY_SOLVER = cvxpy.ECOS

cvxpy.enable_warnings()


def write_test_data(label, value):
    with open(CVXPY_TEST_FILE, "a") as f:
        f.write(f"{label}: \n")
        f.write(pformat(rep_cvxpy(value), width=80))
        f.write("\n\n")


def rep_cvxpy(obj):
    if isinstance(obj, cvxpy.expressions.expression.Expression):
        return obj.name(), obj, rep_cvxpy(obj.value)

    if isinstance(obj, cvxpy.constraints.constraint.Constraint):
        return obj.name(), obj, rep_cvxpy(obj.args)

    if isinstance(obj, cvxpy.problems.objective.Objective):
        return obj.NAME, obj, rep_cvxpy(obj.args)

    if isinstance(obj, dict):
        return {k: rep_cvxpy(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [rep_cvxpy(v) for v in obj]

    return obj


class DAEnergyTimeShift:
    def __init__(self, price: pd.DataFrame, dt: float) -> None:
        self.dt = dt
        self.price = price

    def objective_function(self, variables: Dict[str, Any],
                           generation: cvxpy.Variable = None, annuity_scalar: float = 1.0) -> Dict[str, Any]:
        """Generate the full objective function, including optimization variables.

        :param variables: dictionary of variables being optimized
        :param generation: sum of generation within the system
        :param annuity_scalar: scalar value to be multiplied by any yearly
                cost or benefit that helps capture the cost/benefit over the
                entire project lifetime (only to be set if sizing)

        :return: expression of the objective function that it affects.
                This can be passed into the cvxpy solver.
        """
        p_da = cvxpy.Parameter(self.price.index.size, name='da price', value=[i[0] for i in self.price.values])
        if generation:
            return {'DA ETS': cvxpy.sum(-p_da @ variables['dis'] + p_da @ variables['ch'] -
                                        p_da @ generation) * annuity_scalar * self.dt}
        else:
            return {'DA ETS': cvxpy.sum(-p_da @ variables['dis'] + p_da @ variables['ch']) * annuity_scalar * self.dt}


class PVGen:
    def __init__(self, pv_prod: pd.Series, pv_capacity: float, grid_charge: bool) -> None:
        self.generation = pv_prod
        self.inv_max = pv_capacity
        self.grid_charge = grid_charge
        # self.loc = 'ac'

    def objective_constraints(self, variables: Dict[str, Any], mask: pd.Series) -> list:
        """Build the master constraints for the timeseries data being optimized.

        :param variables: dictionary of variables being optimized
        :param mask: boolean series that is true for indices corresponding to
                time_series data included in the subs data set

        :return: list of constraints that correspond to the battery's physical
                constraints and its service constraints
        """
        constraints = [cvxpy.NonPos(variables['pv_out'] - self.generation[mask])]
        if not self.grid_charge:
            constraints += [cvxpy.NonPos(variables['ch'] - variables['pv_out'])]
        constraints += [
            cvxpy.NonPos(variables['pv_out'] - self.inv_max),
            cvxpy.NonPos(-self.inv_max - variables['pv_out']),
        ]
        return constraints


class BESS:
    def __init__(self, power_capacity: float, energy_capacity: float,
                 rte: float, daily_cycle_limit: float, dt: float, soc_target: float) -> None:
        self.ene_max_rated = energy_capacity
        self.dis_max_rated = power_capacity
        self.ch_max_rated = power_capacity
        self.ulsoc = 1
        self.llsoc = 0
        self.rte = rte
        self.daily_cycle_limit = daily_cycle_limit
        self.dt = dt  # granularity of simulation
        self.soc_target = soc_target

    def objective_constraints(self, variables: Dict[str, Any],
                              mask: pd.Series, reservations: Dict[str, Any]) -> list:
        """Build the master constraint list for the subset of timeseries data being optimized.

        :param variables: dictionary of variables being optimized
        :param mask: boolean series that is true for indices corresponding to
                time_series data included in the subs data set
        :param reservations: dictionary of energy and power reservations
                required by the services being preformed with the current
                optimization subset

        :return: list of constraints that corresponds the battery's physical
                constraints and its service constraints
        """
        ene_target = self.soc_target * self.ulsoc * self.ene_max_rated

        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']

        # create cvx parameters of control constraints (this improves readability in cvx costs and better handling)
        size = int(np.sum(mask))
        ene_max = cvxpy.Parameter(size, name='ene_max', value=np.full((size,), self.ulsoc * self.ene_max_rated))
        ene_min = cvxpy.Parameter(size, name='ene_min', value=np.full((size,), self.llsoc * self.ene_max_rated))
        ch_max = cvxpy.Parameter(size, name='ch_max', value=np.full((size,), self.ch_max_rated))
        ch_min = cvxpy.Parameter(size, name='ch_min', value=np.full((size,), 0.0))
        dis_max = cvxpy.Parameter(size, name='dis_max', value=np.full((size,), self.dis_max_rated))
        dis_min = cvxpy.Parameter(size, name='dis_min', value=np.full((size,), 0.0))

        # energy at the end of the last time step (makes sure that the end of the last time step is ENE_TARGET
        e_res = reservations['E']
        constraints = [
            cvxpy.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - e_res[-1]),

            # energy generally for every time step
            cvxpy.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - e_res[:-1]),

            # energy at the beginning of the optimization window -- handles rolling window
            cvxpy.Zero(ene[0] - ene_target),

            # Keep energy in bounds determined in the constraints configuration function
            # making sure our storage meets control constraints
            cvxpy.NonPos(ene_target - ene_max[-1] + reservations['E_upper'][-1] - variables['ene_max_slack'][-1]),
            cvxpy.NonPos(ene[:-1] - ene_max[:-1] + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1]),

            cvxpy.NonPos(-ene_target + ene_min[-1] + reservations['E_lower'][-1] - variables['ene_min_slack'][-1]),

            cvxpy.NonPos(ene_min[1:] - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1]),

            # Keep charge and discharge power levels within bounds
            cvxpy.NonPos(-ch_max + ch - dis + reservations['D_min'] +
                         reservations['C_max'] - variables['ch_max_slack']),
            cvxpy.NonPos(-ch + dis + reservations['C_min'] +
                         reservations['D_max'] - dis_max - variables['dis_max_slack']),

            # TODO: The following four constraints cause a DDPError warning in cvxpy version 1.1 and later
            cvxpy.NonPos(ch - cvxpy.multiply(ch_max, on_c)),
            cvxpy.NonPos(dis - cvxpy.multiply(dis_max, on_d)),

            # removing the band in between ch_min and dis_min that the battery will not operate in
            cvxpy.NonPos(cvxpy.multiply(ch_min, on_c) - ch + reservations['C_min']),
            cvxpy.NonPos(cvxpy.multiply(dis_min, on_d) - dis + reservations['D_min']),
        ]

        # The constraint below limits energy throughput and total discharge to less than or equal to
        # (number of cycles * energy capacity) per day, for technology warranty purposes
        # this constraint only applies when optimization window is equal to or greater than 24 hours
        days = mask.loc[mask].index.dayofyear
        constraints.extend(
            cvxpy.NonPos(cvxpy.sum(dis[day_mask] * self.dt + cvxpy.pos(e_res[day_mask])) -
                         self.ene_max_rated * self.daily_cycle_limit)
            for day_mask in (day == days for day in days.unique()))
        return constraints


class StorageSolver:
    def __init__(self, cod: int, curve: pd.DataFrame) -> None:
        self.cod = cod
        self.curve = curve
        self.daily_cycle_limit = 2.0
        self.dt = 1.0
        self.grid_charge = False
        self.soc_target = 0.0
        self.rte = 0.87

    def optimization_problem(self, year: int, pv_total_plant: pd.DataFrame, pv_ac_plant: pd.Series,
                             power_capacity: float, energy_capacity: float) -> pd.DataFrame:
        """ Sets up and runs optimization on a subset of data. Called within a loop.

        :return: optimal dispatch variables for each timestep in optimization period.
        """
        # Mask should be a series with length == optimization window
        mask: pd.Series = pv_total_plant > -200
        # Assuming full year window
        size = int(np.sum(mask))

        ##########################################################################
        # COLLECT OPTIMIZATION VARIABLES & POWER/ENERGY RESERVATIONS/THROUGHPUTS #
        ##########################################################################

        # Add optimization variables for each technology
        generation = cvxpy.Variable(shape=size, name='pv_out', nonneg=True)
        variables = {
            'ene': cvxpy.Variable(shape=size, name='ene'),  # Energy at the end of the time step
            'dis': cvxpy.Variable(shape=size, name='dis'),  # Discharge Power, kW during the previous time step
            'ch': cvxpy.Variable(shape=size, name='ch'),  # Charge Power, kW during the previous time step
            'pv_out': generation,
            'ene_max_slack': cvxpy.Parameter(shape=size, name='ene_max_slack', value=np.zeros(size)),
            'ene_min_slack': cvxpy.Parameter(shape=size, name='ene_min_slack', value=np.zeros(size)),
            'dis_max_slack': cvxpy.Parameter(shape=size, name='dis_max_slack', value=np.zeros(size)),
            'dis_min_slack': cvxpy.Parameter(shape=size, name='dis_min_slack', value=np.zeros(size)),
            'ch_max_slack': cvxpy.Parameter(shape=size, name='ch_max_slack', value=np.zeros(size)),
            'ch_min_slack': cvxpy.Parameter(shape=size, name='ch_min_slack', value=np.zeros(size)),
            'on_c': cvxpy.Parameter(shape=size, name='on_c', value=np.ones(size)),
            'on_d': cvxpy.Parameter(shape=size, name='on_d', value=np.ones(size)),
        }

        write_test_data("variables", variables)

        # Calculate system generation
        reservations = {
            'C_max': 0,  # default power and energy reservations
            'C_min': 0,
            'D_max': 0,
            'D_min': 0,
            'E': cvxpy.Parameter(shape=size, value=np.zeros(size), name='zero'),  # energy throughput of a value stream
            'E_upper': cvxpy.Parameter(shape=size, value=np.zeros(size), name='zero'),  # max energy reservation (or throughput if called upon)
            'E_lower': cvxpy.Parameter(shape=size, value=np.zeros(size), name='zero'),  # min energy reservation (or throughput if called upon)
        }

        write_test_data("reservations", reservations)

        #################################################
        # COLLECT OPTIMIZATION CONSTRAINTS & OBJECTIVES #
        #################################################

        da = DAEnergyTimeShift(self.curve[size * year:size * (year + 1)], self.dt)
        expression = da.objective_function(variables, generation)

        write_test_data("expression", expression)

        pv = PVGen(pv_ac_plant, round(max(pv_ac_plant), 0), self.grid_charge)
        bess = BESS(power_capacity, energy_capacity, self.rte,
                    self.daily_cycle_limit, self.dt, self.soc_target)
        constraints = [
            cvxpy.NonPos(-variables['dis'] + variables['ch'] - generation),
            *pv.objective_constraints(variables, mask),
            *bess.objective_constraints(variables, mask, reservations),
        ]

        write_test_data("constraints", constraints)

        objective = cvxpy.Minimize(sum(expression.values()))
        write_test_data("objective", objective)

        variables_ = objective.variables()
        write_test_data("variables", variables_)

        prob = cvxpy.Problem(objective, constraints)

        constants = prob.constants()
        parameters = prob.parameters()
        write_test_data("constants", constants)
        write_test_data("parameters", parameters)

        problem_data = prob.get_problem_data(CVXPY_SOLVER)
        write_test_data("problem_data", problem_data)

        start = time.perf_counter()
        prob.solve(solver=CVXPY_SOLVER, verbose=CVXPY_VERBOSE)  # Allow user to specify solver
        end = time.perf_counter()
        if CVXPY_VERBOSE:
            print(f'{CVXPY_SOLVER or "Default"} solver finished in {end - start} seconds.')
            print('Optimization problem was', prob.status)
        if prob.status.split(maxsplit=1)[0] != 'optimal':
            raise cvxpy.SolverError(f'Problem is {prob.status}; no solution found')

        ################################################
        # POST-OPTIMIZATION: COLLECT RESULTS TO RETURN #
        ################################################

        variable_values = pd.DataFrame({name: value.value for name, value in variables.items()}, index=mask.index)
        return variable_values


if __name__ == "__main__":
    data = pd.read_csv('test_data.csv', index_col=0)
    data = data['Values']
    data.index = pd.to_datetime(data.index)
    cod = datetime.date(2021, 3, 1)
    curve = pd.read_csv("curve.csv")
    solver = StorageSolver(cod, curve)
    solution = solver.optimization_problem(0, data, data, 100, 400)


