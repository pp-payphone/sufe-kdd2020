"""km algorithm"""


class bfsKM(object):
    """广度优先km算法"""
    def __init__(self, N, weight):
        self.N = N
        self.weight = weight
        self.left_match, self.right_match = [-1] * N, [-1] * N  # -1表示未被匹配
        self.left_ver, self.right_ver = [0] * N, [0] * N
        for j in range(N):
            self.left_ver[j] = max(weight[j])
        self.que = [-1] * (2*N)
        self.pre = [-1] * N

    def check(self, i):
        self.right_vis[i] = True
        if self.right_match[i] != -1:
            self.que[self.tail] = self.right_match[i]
            self.tail += 1
            self.left_vis[self.right_match[i]] = True
            return False
        while i != -1:
            self.right_match[i] = self.pre[i]
            tmp = self.left_match[self.pre[i]]
            self.left_match[self.pre[i]] = i
            i = tmp
        return True

    def bfs(self, S):
        self.slack = [float('inf')] * self.N
        self.left_vis, self.right_vis = [False] * self.N, [False] * self.N
        self.head, self.tail = 0, 1
        self.que[0] = S
        self.left_vis[S] = True
        while True:
            while self.head < self.tail:
                j = self.que[self.head]
                self.head += 1
                for i in range(self.N):
                    diff = self.right_ver[i] + self.left_ver[j] - self.weight[j][i]
                    if not self.right_vis[i] and self.slack[i] >= diff:
                        self.pre[i] = j
                        if diff:
                            self.slack[i] = diff
                        elif self.check(i):
                            return
            delta = float('inf')
            for i in range(self.N):
                if not self.right_vis[i] and delta > self.slack[i]:
                    delta = self.slack[i]
            for i in range(self.N):
                if self.right_vis[i]:
                    self.right_ver[i] += delta
                else:
                    self.slack[i] -= delta
                if self.left_vis[i]:
                    self.left_ver[i] -= delta
            for i in range(self.N):
                if not self.right_vis[i] and self.slack[i] == 0 and self.check(i):
                    return
