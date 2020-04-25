"""agent."""
from bfskm import bfsKM
import numpy as np
from gridsearch import GridSearch
import os


class Agent(object):
    """Agent for dispatching and reposition"""

    def __init__(self):
        self.value_fun = np.load(os.path.join(os.path.dirname(__file__), "mdp_value_5.npy"))
        self.gamma = 0.95
        self.BASETIME = 946670400  # 2020.1.1 04:00:00
        self.dispatch_frequency_gap = 300  # 以5分钟为时间间隔
        self.grid_search = GridSearch()
        # self.gridsearch = GridSearch()
        pass

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
        order_ids, driver_ids, result = self.process(dispatch_observ)
        num_orders, num_drivers = len(order_ids), len(driver_ids)
        dispatch_action = []
        left_num, right_num = num_drivers, num_orders  # 司机放左边，订单放右边
        N = max(left_num, right_num)
        weight = [[0] * N for i in range(N)]
        for j in range(left_num):
            for i in range(right_num):
                if (order_ids[i], driver_ids[j]) in result:
                    weight[j][i] = result[(order_ids[i], driver_ids[j])]
        km = bfsKM(N, weight)
        for j in range(N):
            km.bfs(j)
        for i in range(right_num):
            if km.weight[km.right_match[i]][i] != 0:
                dispatch_action.append(dict(order_id=order_ids[i], driver_id=driver_ids[km.right_match[i]]))
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

    def process(self, dispatch_observ):
        order_ids = set()
        driver_ids = set()
        result = dict()
        for od in dispatch_observ:
            oid, did, fts, lts = od["order_id"], od["driver_id"], od["timestamp"], od["order_finish_timestamp"] + od["pick_up_eta"]
            fgrid, lgrid = self.grid_search.cal_loc_grid([od["driver_location"], od["order_finish_location"]])
            ftid = int(((fts - self.BASETIME)//self.dispatch_frequency_gap) % (24*3600//self.dispatch_frequency_gap))
            ltid = int(((lts - self.BASETIME)//self.dispatch_frequency_gap) % (24*3600//self.dispatch_frequency_gap))
            order_ids.add(oid)
            driver_ids.add(did)
            discount_rate = pow(self.gamma, ltid-ftid)
            result[(oid, did)] = od["reward_units"] * discount_rate + discount_rate * self.value_fun[ltid][lgrid] - self.value_fun[ftid][fgrid]
        order_ids = list(order_ids)
        driver_ids = list(driver_ids)
        return order_ids, driver_ids, result
