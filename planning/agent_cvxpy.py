"""agent."""
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import pickle
import time
# import timeit
from grid_search import GridSearch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx


class Agent(object):
    """Agent for dispatching and reposition"""

    def __init__(self):
        with open(os.path.join(dir_path, "value_function.pkl"), "rb") as f:
            self.v_f = pickle.load(f)
        self.gamma = 0.9
        self.advantage_fun = None
        self.gridsearch = GridSearch()

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
            order_id, int
            driver_id, int
            order_driver_distance, float
            order_start_location, a list as [lng, lat], float
            order_finish_location, a list as [lng, lat], float
            driver_location, a list as [lng, lat], float
            timestamp, int
            order_finish_timestamp, int
            day_of_week, int
            reward_units, float
            pick_up_eta, float

        :return: a list of dict, the key in the dict includes:
            order_id and driver_id, the pair indicating the assignment
        """
        # 做driver_id映射
        order_set = set()
        driver_set = set()
        for od in dispatch_observ:
            order_set.add(od["order_id"])
            driver_set.add(od["driver_id"])
        driver_id_dict = dict(zip(driver_set, range(len(driver_set))))
        # driver_id_refresh_dict = dict(zip(range(len(driver_set)), driver_set))

        # 计算advantage_fun
        self.advantage_fun = pd.DataFrame(np.zeros([len(driver_set), len(order_set)]), columns=list(order_set), index=driver_id_dict.values())
        for od in dispatch_observ:
            oid, did = od["order_id"], driver_id_dict[od["driver_id"]]
            fgrid, lgrid = self.gridsearch.cal_loc_grid([od["driver_location"], od["order_finish_location"]])
            ftid = self.cal_time_index_for_day(od["timestamp"])
            ltid = self.cal_time_index_for_day(od["order_finish_timestamp"])
            self.advantage_fun[oid][did] = pow(self.gamma, ltid-ftid)*self.v_f[ltid][lgrid] - self.v_f[ftid][fgrid] + od["reward_units"]

        # 标准化输入格式
        rec_num = len(dispatch_observ)
        order_mat_row, order_mat_col = [None] * rec_num, [None] * rec_num
        driver_mat_row, driver_mat_col = [None] * rec_num, [None] * rec_num
        order_idx_gmv = np.zeros(rec_num)
        order_num, driver_num = 0, 0
        for idx, od in enumerate(dispatch_observ):
            oid, did = od["order_id"], driver_id_dict[od["driver_id"]]
            order_num = max(order_num, oid + 1)
            driver_num = max(driver_num, did + 1)
            order_idx_gmv[idx] = self.advantage_fun[oid][did]
            order_mat_row[idx], order_mat_col[idx] = oid, idx
            driver_mat_row[idx], driver_mat_col[idx] = did, idx
        order_mat = sp.coo_matrix((np.ones(rec_num), (order_mat_row, order_mat_col)), shape=(order_num, rec_num))
        driver_mat = sp.coo_matrix((np.ones(rec_num), (driver_mat_row, driver_mat_col)), shape=(driver_num, rec_num))

        # cvxpy求解凸优化问题
        X = cvx.Variable(rec_num, boolean=True)
        obj = order_idx_gmv*X
        constr = [
            order_mat * X <= 1,
            driver_mat * X <= 1,
        ]
        prob = cvx.Problem(cvx.Maximize(obj), constr)
        prob.solve(solver=cvx.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF', 'presolve': 'GLP_ON'})
        # opt_v = prob.value
        opt_X = X.value

        # 输出格式标准化
        dispatch_action = []
        for idx, od in enumerate(dispatch_observ):
            if opt_X[idx] == 1:
                oid, did = od["order_id"], od["driver_id"]
                dispatch_action.append(dict(order_id=oid, driver_id=did))
        return dispatch_action

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
            timestamp: int
            driver_info: a list of dict, the key in the dict includes:
                driver_id: driver_id of the idle driver in the treatment group, int
                grid_id: id of the grid the driver is located at, str
            day_of_week: int

        :return: a list of dict, the key in the dict includes:
            driver_id: corresponding to the driver_id in the od_list
            destination: id of the grid the driver is repositioned to, str
        """
        repo_action = []
        for driver in repo_observ['driver_info']:
        # the default reposition is to let drivers stay where they are
            repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action

    def cal_time_index_for_day(self, timestamp, dispatch_freqency_gap=300, BASEHOUR=0) -> int:
        """
        args:绝对时间  tid间隔，默认5m  tid=0对应当天时间，默认0:00
        return:tid
        """
        ts = time.localtime(timestamp)
        tid = tid = ((ts[3]-BASEHOUR)*3600 + ts[4]*60 + ts[5])//300
        return tid
