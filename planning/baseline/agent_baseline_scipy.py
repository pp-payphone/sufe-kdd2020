"""agent."""
from scipy.optimize import linear_sum_assignment
import numpy as np


class Agent(object):
    """Agent for dispatching and reposition"""

    def __init__(self):
        self.gamma = 0.95
        self.BASETIME = 946670400  # 2020.1.1 04:00:00
        self.dispatch_frequency_gap = 300  # 以5分钟为时间间隔
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
        driver_id_refresh_dict, result = self.process(dispatch_observ)
        dispatch_action = []
        row_ind, col_ind = linear_sum_assignment(result)
        for idx, oid in enumerate(row_ind):
            dispatch_action.append(dict(order_id=oid, driver_id=driver_id_refresh_dict[col_ind[idx]]))
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
        for od in dispatch_observ:
            order_ids.add(od["order_id"])
            driver_ids.add(od["driver_id"])
        driver_id_dict = dict(zip(driver_ids, range(len(driver_ids))))
        driver_id_refresh_dict = dict(zip(range(len(driver_ids)), driver_ids))
        result = np.zeros([len(order_ids), len(driver_ids)])
        for od in dispatch_observ:
            oid, did, fts, lts = od["order_id"], driver_id_dict[od["driver_id"]], od["timestamp"], od["order_finish_timestamp"] + od["pick_up_eta"]
            ftid = int(((fts - self.BASETIME)//self.dispatch_frequency_gap) % (24*3600//self.dispatch_frequency_gap))
            ltid = int(((lts - self.BASETIME)//self.dispatch_frequency_gap) % (24*3600//self.dispatch_frequency_gap))
            order_ids.add(oid)
            driver_ids.add(did)
            tid_period = ltid - ftid
            if tid_period == 0:
                result[oid, did] = od["reward_units"]
            else:
                result[oid, did] = (od["reward_units"]*(1-pow(self.gamma, tid_period)))/(tid_period*(1-self.gamma))
        return driver_id_refresh_dict, result
