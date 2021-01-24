import numpy as np
from scipy import spatial as ss

# this is borrowed from https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization

# Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign
    
    
    
def compute_tp(pred,gt):
    
    pred_p_list = np.nonzero(pred)
    gt_p_list = np.nonzero(gt)
    pred_num = len(pred_p_list[0])
    gt_num = len(gt_p_list[0])
    pred_p = np.zeros([pred_num,2])
    gt_p = np.zeros([gt_num,2])
    pred_p[:,0] = pred_p_list[0]
    pred_p[:,1] = pred_p_list[1]       
    gt_p[:,0] = gt_p_list[0]
    gt_p[:,1] = gt_p_list[1]
           
    dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
    match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]    
        match_matrix[i_pred_p,:] = pred_dist<=20

    tp, assign = hungarian(match_matrix)
    
    return tp
