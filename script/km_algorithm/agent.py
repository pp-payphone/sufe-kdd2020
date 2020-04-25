class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
    """ Load your trained model and initialize the parameters """
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

    def process(self, dispatch_observ):
        order_ids = set()
        driver_ids = set()
        result = dict()
        for od in dispatch_observ:
            oid, did = od["order_id"], od["driver_id"]
            order_ids.add(oid)
            driver_ids.add(did)
            result[(oid, did)] = od["reward_units"]
        order_ids = list(order_ids)
        driver_ids = list(driver_ids)
        return order_ids, driver_ids, result