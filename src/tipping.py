from joblib import Parallel, delayed
import multiprocessing

n_jobs = multiprocessing.cpu_count()

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import spearmanr
import os
import json
from climateforcing.utils import mkdir_p
import xarray as xr
from itertools import product
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import math
from tqdm import tqdm
import itertools
from itertools import combinations

import re

from os import listdir
from os.path import isfile, join

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from pycascades.core.coupling import coupling
from pycascades.core.tipping_element import cusp

import networkx as nx
from tqdm import trange

# shared LHS tipping parameters (single source of truth; used here + by the
# feedback analysis + the calculator, so sample indices stay aligned)
from tipping_params import param_bounds, pf_bounds, sample_lhs_params

from scipy.integrate import quad

# ── cusp timescale conversion (pycascades earth_system/timing.py; Wunderling et al. 2021) ──
# The literature tipping timescale tau is the time an UNCOUPLED cusp needs to go from x=-1 to
# x=+1 at GMT_CAL with threshold TCRIT_CAL. In dx/dt = (1/T)(-x^3 + x + c) that transition takes
# I_CAL * T, so the ODE time constant must be T = tau / I_CAL (else every element is I_CAL too slow).
GMT_CAL, TCRIT_CAL = 4.0, 1.8
I_CAL = quad(lambda x: 1.0 / (-x**3 + x + np.sqrt(4 / 27) * GMT_CAL / TCRIT_CAL), -1.0, 1.0)[0]  # 2.6267


def ode_timescale(tau):
    """Literature tipping time tau (yr) -> ODE time constant T (yr) of the cusp."""
    return tau / I_CAL


class global_functions():
    """
    Linear feedback function to compute feedbacks. Maximal feedback is obtained from 2.0°C onwards.
    feedbacks for Arctic summer sea ice, mountain glaciers, GIS and WAIS are proven to be constant also for higher temperatures
    feedbacks for Amazon and feedbacks_steffen are computed for a temperature increase of 2.0°C until 2100, see paper
    """
    def feedback_function(GMT, fbmax):
        # fbmax = maximal feedback
        # only returns a value in case GMT is higher than lower boundary of the respective tipping element, otherwise return 0.0 (lower cap), N.B.: No upper cap
        if GMT <= 2.0:
            y = (fbmax / 2.0) * GMT
            return y
        elif GMT > 2.0:
            return fbmax
        else:
            raise Exception("GMT negativ: Feedbacks do not work for temperatures smaller 0!")

    # feedbacks of state dependent variables, linear increase of feedbacks between state -1 and +1 for GIS, WAIS, THC, NINO and AMAZ
    def state_feedback(state, fbmax):
        if state >= -1 and state <= 1:
            y = fbmax / 2 * (state + 1)
        elif state < -1:
            y = 0.
        elif state > +1:
            y = fbmax
        return y

    # c = c(GMT) where tipping occurs at sqrt(4/27) ~ 0.38
    # Linear function through two points maps GMT --> c, where x-values represent GMT and y-values represent CUSP-c values
    def CUSPc(x1, x2, x):
        # only returns a value in case GMT is higher than lower boundary of the respective tipping element, otherwise return 0.0 (lower cap), N.B.: No upper cap
        if x >= x1:
            y1 = 0.0
            y2 = np.sqrt(4 / 27)
            y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
            return y
        else:
            return 0.0

class linear_coupling_earth_system(coupling):

    def __init__(self, strength, x_0):
        coupling.__init__(self)
        self._strength = strength
        self._x_0 = x_0

    def dxdt_cpl(self):
        return lambda t, x_from, x_to: self._strength * (x_from - self._x_0)

    def jac_cpl(self):
        return lambda t, x_from, x_to: self._strength

    def jac_diag(self):
        return lambda t, x_from, x_to: 0

    def bif_impact(self):
        return lambda t, x_from, x_to: self._strength * (x_from - self._x_0)

class tipping_network(nx.DiGraph):

    def __init__( self, incoming_graph_data=None, **attr):
        nx.DiGraph.__init__( self, incoming_graph_data=None, **attr)

    def add_element( self, tipping_element ):
        ind = self.number_of_nodes()
        super().add_node( ind, data = tipping_element )
        self.nodes[ind]['lambda_f'] = tipping_element.dxdt_diag()
        self.nodes[ind]['lambda_jac'] = tipping_element.jac_diag()

    def add_coupling( self, from_id, to_id, coupling):
        super().add_edge( from_id, to_id, data = coupling)
        self[from_id][to_id]['lambda_f'] = coupling.dxdt_cpl()
        self[from_id][to_id]['lambda_jac'] = coupling.jac_cpl()
        self[from_id][to_id]['lambda_jac_diag'] = coupling.jac_diag()

    def set_param( self, node_id, key, val ):
        element = self.nodes[node_id]['data']
        element.set_par( key, val)
        self.nodes[node_id]['lambda_f'] = self.nodes[node_id]['data'].dxdt_diag()
        self.nodes[node_id]['lambda_jac'] = self.nodes[node_id]['data'].jac_diag()

    def get_tip_states( self, x):
        tipped = [self.nodes[i]['data'].tip_state()(x[i]) for i in self.nodes()]
        return np.array( tipped )

    def get_node_types( self ):
        type_list = [self.nodes[i]['data'].get_type() for i in self.nodes()]
        return type_list

    def get_number_tipped( self, x):
        return np.count_nonzero( self.get_tip_states( x ) )

    def f( self, x, t):
        f = np.zeros( self.number_of_nodes() )
        for node in self.nodes(data=True):
            ind = node[0]
            x_comp = x[ind]
            f[ind] = node[1]['lambda_f'].__call__( t, x_comp)
        for edge in self.edges(data=True):
            from_id = edge[0]
            to_id = edge[1]
            lmd = edge[2]['lambda_f']
            f[to_id] += lmd.__call__( t, x[from_id], x[to_id])
        return f

    def jac(self, x, t):
        jac = np.zeros((self.number_of_nodes(), self.number_of_nodes()))
        for node in self.nodes(data=True):
            ind = node[0]
            x_comp = x[ind]
            jac[ind,ind] = node[1]['lambda_jac'].__call__( t, x_comp)
        for edge in self.edges(data=True):
            from_id = edge[0]
            to_id = edge[1]
            lmd = edge[2]['lambda_jac']
            jac[to_id, from_id] = lmd.__call__( t, x[from_id], x[to_id] )
            lmd_diag = edge[2]['lambda_jac_diag']
            jac[to_id, to_id] += lmd_diag.__call__( t, x[from_id], x[to_id] )
        return jac

    def set_vulnerability(self, node_id, bool_val):
        self.nodes[node_id]["vulnerable"] = bool_val

    def get_vulnerability_network(self):
        G = nx.Graph()
        for node in self.nodes():
            G.add_node(node)
            G.nodes[node]["vulnerable"] = self.nodes[node]["vulnerable"]
        for edge in self.edges():
                if self.nodes[edge[0]]["vulnerable"] and self.nodes[edge[1]]["vulnerable"]:
                    G.add_edge(edge[0],edge[1])
        return G

    def get_out_component_size(self, from_id):
        out_component_size = -1
        for node in self.nodes():
            if nx.has_path(self, from_id, node):
                out_component_size += 1
        return out_component_size

    def compute_impact_matrix(self):
        n = self.number_of_nodes()
        impact_matrix = [[lambda t, x1, x2: 0 for j in range(n)] for i in range(n)]
        for edge in self.edges.data():
            print(edge[2]['data'].bif_impact())
            impact_matrix[edge[1]][edge[0]]=edge[2]['data'].bif_impact()
            # if self.get_node_types()[edge[1]]=='cusp':
            #     impact_matrix[edge[1]][edge[0]]=edge['data'].dxdt_cpl()
            # elif self.get_node_types()[edge[1]]=='hopf':
            #     impact_matrix[edge[1]][edge[0]]=edge['data']
        return impact_matrix

# The literature timescales tau (gis_time, ...) are converted once to the cusp ODE time
# constant T = tau / I_CAL (pycascades earth_system/timing.py; Wunderling et al. 2021). Because
# every term — the cusp (a,b,c) and all couplings — carries the same 1/T, the conversion only
# rescales the clock; equilibria and coupling ratios are unchanged. convert_tau=False keeps the
# raw tau (the old, I_CAL-too-slow behaviour) for comparison only.
class Earth_System():
    def __init__(self, gis_time, thc_time, wais_time, amaz_time, limits_gis, limits_thc, limits_wais, limits_amaz,
                  pf_wais_to_gis, pf_thc_to_gis, pf_gis_to_thc, pf_wais_to_thc, pf_gis_to_wais, pf_thc_to_wais, pf_thc_to_amaz,
                  convert_tau=True):
        #timescales (literature tau -> ODE time constant T = tau / I_CAL)
        _conv = ode_timescale if convert_tau else (lambda t: t)
        self._gis_time = _conv(gis_time)
        self._thc_time = _conv(thc_time)
        self._wais_time = _conv(wais_time)
        self._amaz_time = _conv(amaz_time)

        #tipping limits
        self._limits_gis = limits_gis
        self._limits_thc = limits_thc
        self._limits_wais = limits_wais
        self._limits_amaz = limits_amaz

        #probability fractions
        self._pf_wais_to_gis = pf_wais_to_gis
        self._pf_thc_to_gis = pf_thc_to_gis
        self._pf_gis_to_thc = pf_gis_to_thc
        self._pf_wais_to_thc = pf_wais_to_thc
        self._pf_gis_to_wais = pf_gis_to_wais
        self._pf_thc_to_wais = pf_thc_to_wais
        self._pf_thc_to_amaz = pf_thc_to_amaz

    """
    you must provide this method with a global mean temperature, a coupling strength and
    an integer (-1, 0, +1) for the network type that you want to invoke, i.e. kk0, kk1 and kk2 must be -1, 0 or +1
    """
    def earth_network(self, effective_GMT, strength, kk0, kk1):
        gis = cusp(a=-1 / self._gis_time, b=1 / self._gis_time, c=(1 / self._gis_time) * global_functions.CUSPc(0., self._limits_gis, effective_GMT), x_0=0.0)
        thc = cusp(a=-1 / self._thc_time, b=1 / self._thc_time, c=(1 / self._thc_time) * global_functions.CUSPc(0., self._limits_thc, effective_GMT), x_0=0.0)
        wais = cusp(a=-1 / self._wais_time, b=1 / self._wais_time, c=(1 / self._wais_time) * global_functions.CUSPc(0., self._limits_wais, effective_GMT), x_0=0.0)
        amaz = cusp(a=-1 / self._amaz_time, b=1 / self._amaz_time, c=(1 / self._amaz_time) * global_functions.CUSPc(0., self._limits_amaz, effective_GMT), x_0=0.0)

        # set up network
        net = tipping_network()
        net.add_element(gis)
        net.add_element(thc)
        net.add_element(wais)
        net.add_element(amaz)


        ######################################Set edges to active state#####################################
        net.add_coupling(1, 0, linear_coupling_earth_system(strength=-(1 / self._gis_time) * strength * self._pf_thc_to_gis, x_0=-1))
        net.add_coupling(2, 0, linear_coupling_earth_system(strength=(1 / self._gis_time) * strength * self._pf_wais_to_gis, x_0=-1))

        net.add_coupling(0, 1, linear_coupling_earth_system(strength=(1 / self._thc_time) * strength * self._pf_gis_to_thc, x_0=-1))
        net.add_coupling(2, 1, linear_coupling_earth_system(strength=(1 / self._thc_time) * strength * self._pf_wais_to_thc * kk0, x_0=-1))

        net.add_coupling(0, 2, linear_coupling_earth_system(strength=(1 / self._wais_time) * strength * self._pf_gis_to_wais, x_0=-1))
        net.add_coupling(1, 2, linear_coupling_earth_system(strength=(1 / self._wais_time) * strength * self._pf_thc_to_wais, x_0=-1))

        net.add_coupling(1, 3, linear_coupling_earth_system(strength=(1 / self._amaz_time) * strength * self._pf_thc_to_amaz * kk1, x_0=-1))

        return net

    def dynamic_earth_network(self, gmt_function, strength, kk0, kk1):
        gis = cusp(a=-1 / self._gis_time, b=1 / self._gis_time, c= lambda t : (1 / self._gis_time) * global_functions.CUSPc(0., self._limits_gis, gmt_function(t)), x_0=0.0)
        thc = cusp(a=-1 / self._thc_time, b=1 / self._thc_time, c=lambda t :(1 / self._thc_time) * global_functions.CUSPc(0., self._limits_thc, gmt_function(t)), x_0=0.0)
        wais = cusp(a=-1 / self._wais_time, b=1 / self._wais_time, c=lambda t :(1 / self._wais_time) * global_functions.CUSPc(0., self._limits_wais, gmt_function(t)), x_0=0.0)
        amaz = cusp(a=-1 / self._amaz_time, b=1 / self._amaz_time, c=lambda t :(1 / self._amaz_time) * global_functions.CUSPc(0., self._limits_amaz, gmt_function(t)), x_0=0.0)

        # set up network
        net = tipping_network()
        net.add_element(gis)
        net.add_element(thc)
        net.add_element(wais)
        net.add_element(amaz)


        ######################################Set edges to active state#####################################
        net.add_coupling(1, 0, linear_coupling_earth_system(strength=-(1 / self._gis_time) * strength * self._pf_thc_to_gis, x_0=-1))
        net.add_coupling(2, 0, linear_coupling_earth_system(strength=(1 / self._gis_time) * strength * self._pf_wais_to_gis, x_0=-1))

        net.add_coupling(0, 1, linear_coupling_earth_system(strength=(1 / self._thc_time) * strength * self._pf_gis_to_thc, x_0=-1))
        net.add_coupling(2, 1, linear_coupling_earth_system(strength=(1 / self._thc_time) * strength * self._pf_wais_to_thc * kk0, x_0=-1))

        net.add_coupling(0, 2, linear_coupling_earth_system(strength=(1 / self._wais_time) * strength * self._pf_gis_to_wais, x_0=-1))
        net.add_coupling(1, 2, linear_coupling_earth_system(strength=(1 / self._wais_time) * strength * self._pf_thc_to_wais, x_0=-1))

        net.add_coupling(1, 3, linear_coupling_earth_system(strength=(1 / self._amaz_time) * strength * self._pf_thc_to_amaz * kk1, x_0=-1))

        return net


# ── importable tipping computation (extracted verbatim from the per-scenario run) ──
elements = ["GIS", "THC", "WAIS", "AMAZ"]


def compute_tip_prob(temperature_da, params, n_samples, configs, runs,
                     t_start=0, t_end=15000, n_eval=1001, n_jobs=n_jobs, convert_tau=True):
    """Tipping probabilities for one scenario's temperature field.

    Parameters
    ----------
    temperature_da : xr.DataArray
        Temperature for a single scenario (dims include config, run, timebounds).
    params : dict
        LHS tipping parameters (from ``sample_lhs_params``); arrays of length >= n_samples.
    n_samples : int
        Number of LHS samples to run (uses the first n_samples of each parameter array).
    configs, runs : list
        config / run coordinate values to iterate over.

    Returns (prob_any_tipping, prob_any_tipping_sample, prob_element_tipping)
    DataArrays.
    """
    t_eval = np.linspace(t_start, t_end, n_eval)

    def run_single_sample(i, gmt_function):

        pr_i = {key: params[key][i] for key in param_bounds}
        pf_i = {key: params[key][i] for key in pf_bounds}

        sys = Earth_System(
            **pr_i,
            **pf_i,
            convert_tau=convert_tau
        )

        net = sys.dynamic_earth_network(
            gmt_function=gmt_function,
            strength=params["strength"][i],
            kk0=1,
            kk1=1
        )
        sol = solve_ivp(
            fun=lambda t, x: net.f(x, t),
            t_span=(t_start, t_end),
            y0=[-1, -1, -1, -1],
            t_eval=t_eval,
            atol=1e-3,
            rtol=1e-3,
            method="LSODA"
        )
        return np.any(sol.y > 0.0, axis=1)

    prob_any = xr.DataArray(
        np.nan,
        dims=("config", "run"),
        coords=dict(config=configs, run=runs),
        name="prob_any_tipping"
    )
    prob_any_sample = xr.DataArray(
        np.nan,
        dims=("sample", "config", "run"),
        coords=dict(sample=np.arange(n_samples), config=configs, run=runs),
        name="prob_any_tipping_sample"
    )
    prob_elements = xr.DataArray(
        np.nan,
        dims=("config", "run", "element"),
        coords=dict(config=configs, run=runs, element=elements),
        name="prob_element_tipping"
    )

    for config in tqdm(configs, desc="Config"):
        for run in runs:

            # --- extract temperature trajectory
            temp = temperature_da.sel(config=config, run=run)

            # get temp and time to be 1D array like for interp1d
            time = temp.timebounds.values
            temp = temp.values

            # Drop NaN before building the interpolant. The concatenated temperature
            # spans the common horizon (SSPs to 2300, NGFS/branch-offs only to 2110),
            # so shorter scenarios are NaN-padded to 2300 and even the SSPs carry a
            # NaN at the final bound. With NaN in `temp`, fill_value=(temp[0], temp[-1])
            # feeds NaN to the cascade for the long hold-out (t up to 15000) -> the
            # whole integration becomes NaN and NO element ever registers as tipped.
            # Using only valid points holds the last *valid* committed temperature.
            valid = ~np.isnan(temp)
            time = time[valid]
            temp = temp[valid]

            gmt_function = interp1d(
                time,
                temp,
                kind="linear",
                bounds_error=False,
                fill_value=(temp[0], temp[-1])
            )

            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_sample)(i, gmt_function) for i in range(n_samples)
            )
            tipped = np.vstack(results)

            # --- probabilities
            prob_any.loc[config, run] = np.any(tipped, axis=1).mean()
            prob_any_sample.loc[:, config, run] = np.any(tipped, axis=1)
            prob_elements.loc[config, run, :] = tipped.mean(axis=0)

    return prob_any, prob_any_sample, prob_elements
