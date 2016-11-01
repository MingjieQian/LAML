package org.laml.ml.graph;

import java.util.LinkedList;
import java.util.List;

/**
 * All-pairs shortest paths.
 * 
 * @author Mingjie Qian
 * @version 1.0 April 23rd, 2016
 */
public class AllPairsShortestPaths {

	public static void main(String[] args) {

	}
	
	/**
	 * <p>
	 * The Floyd-Warshall algorithm relies on the following observation. 
	 * Under our assumption that the vertices of G are V = {1, 2, ..., n}, 
	 * let us consider a subset {1, 2, ..., k} of vertices for some k. 
	 * For any pair of vertices i, j \in V , consider all paths from i to 
	 * j whose intermediate vertices are all drawn from {1, 2, ..., k}, 
	 * and let p be a minimum-weight path from among them. (Path p is 
	 * simple.) The Floyd-Warshall algorithm exploits a relationship 
	 * between path p and shortest paths from i to j with all 
	 * intermediate vertices in the set {1, 2, ..., k-1}. The relationship
	 * depends on whether or not k is an intermediate vertex of path p.
	 * 
	 * <p>
	 * The recursion is d_ij(k) = min(d_ij(k-1), d_ik(k-1) + d_kj(k-1)). 
	 * Assume that there are no negative cycles.
	 * Let j = k, we have d_ik(k) = d_ik(k-1).
	 * Let i = k, we have d_kj(k) = d_kj(k-1).
	 * Now when i < k, d_ij(k) = min(d_ij(k-1), d_ik(k-1) + d_kj(k-1)), 
	 * we just use old entries of k-th row: d_kj(k-1).
	 * Now when i = k, d_ij(k) = min(d_ij(k-1), d_ik(k-1) + d_kj(k-1)), 
	 * we don't need to update. Even we compute, the result will always 
	 * be the same: d_ij(k) = d_ij(k-1).
	 * Now when i > k, d_ij(k) = min(d_ij(k-1), d_ik(k-1) + d_kj(k)), 
	 * we just use new entries of k-th row: d_kj(k).
	 * When j is traversed from 1 to n, since d_ik(k) = d_ik(k-1) is 
	 * unchanged and d_ij(k) = min(d_ij(k-1), ...), we use old entries 
	 * of i-th row: d_ij(k-1).
	 * 
	 * <p>
	 * When there is at lease one negative cycle, we cannot simply drop 
	 * all the superscripts.
	 * 
	 * <p>
	 * Since π_ij(k) = π_kj(k-1) if d_ij(k-1) > d_ik(k-1) + d_kj(k-1), 
	 * π_kj(k-1) = π_kj(k),
	 * when i < k, we use old entries π_kj(k-1),
	 * when i = k, π_kj(k) is always π_kj(k-1) no matter if there're 
	 * negative cycles (d_kk(k-1) < 0 or not),
	 * when i > k, we use new entries π_kj(k).
	 * 
	 * <p>
	 * Note that π_ij is the predecessor of j in the shortest simple 
	 * from i to j.
	 * 
	 * <p>
	 * W[i][j] = 0 if i == j or w(i, j) if i -> j \in E or +\infty 
	 * otherwise.
	 * 
	 * @param W
	 * @return
	 */
	public static double[][] FloydWarshall(double[][] W) {
		int n = W.length;
		double[][] D = W.clone();
		int[][] P = new int[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j || Double.isInfinite(W[i][j]))
					P[i][j] = -1;
				else
					P[i][j] = i;
			}
		}
		for (int k = 0; k < n; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					double s = D[i][k] + D[k][j];
					if (D[i][j] > s) {
						D[i][j] = s;
						P[i][j] = P[k][j];
					}
				}
			}
		}
		@SuppressWarnings("unchecked")
		List<Integer>[] paths = new List[n * n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int idx = i * n + j;
				List<Integer> path = new LinkedList<Integer>();
				if (Double.isInfinite(D[i][j])) {
					paths[idx] = path;
					continue;
				}
				int t = j;
				do {
					path.add(0, t);
					t = P[i][t];
				} while (i != t);
				path.add(0, i);
				paths[idx] = path;
			}
		}
		return D;
	}

}
