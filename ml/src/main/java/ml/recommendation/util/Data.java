package ml.recommendation.util;

import java.util.HashMap;
import java.util.LinkedList;

import la.vector.Vector;

public class Data {

	public Data(
			int M, 
			int N, 
			int T, 
			double[] Yij, 
			double[][] Xij, 
			int[] UserIndices,
			int[] ItemIndices,
			HashMap<Integer, LinkedList<Integer>> CUser,
			HashMap<Integer, LinkedList<Integer>> CItem,
			HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap,
			HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap,
			Vector[] Xi,
			Vector[] Xj,
			int Pu,
			int Pv,
			int Pe
			) {
		this.M = M;
		this.N = N;
		this.T = T;
		this.Yij = Yij;
		this.Xij = Xij;
		this.UserIndices = UserIndices;
		this.ItemIndices = ItemIndices;
		this.CUser = CUser;
		this.CItem = CItem;
		this.User2EventIndexSetMap = User2EventIndexSetMap;
		this.Item2EventIndexSetMap = Item2EventIndexSetMap;
		this.Xi = Xi;
		this.Xj = Xj;
		this.Pu = Pu;
		this.Pv = Pv;
		this.Pe = Pe;
	}

	public Data() {
	}

	/**
	 * Number of users.
	 */
	public int M = 0;
	
	/**
	 * Number of items.
	 */
	public int N = 0;
	
	/**
	 * Number of events.
	 */
	public int T = 0;
	
	/**
	 * User feature size.
	 */
	public int Pu = 0;
	
	/**
	 * Item feature size.
	 */
	public int Pv = 0;
	
	/**
	 * Event feature size.
	 */
	public int Pe = 0;
	
	/**
	 * Array of labels for all events.
	 */
	public double[] Yij = null;
	
	/**
	 * Array of feature vectors for all events.
	 */
	// public Vector[] XijVectors = null;
	
	/**
	 * Array of feature vectors for all events.
	 */
	public double[][] Xij = null;
	
	/**
	 * UserIndices[k] is the user index for the k-th event.
	 */
	public int[] UserIndices = null;
	
	/**
	 * ItemIndices[k] is the item index for the k-th event.
	 */
	public int[] ItemIndices = null;
	
	/**
	 * CUser[i] = {j|(i, j) \in C}.
	 */
	public HashMap<Integer, LinkedList<Integer>> CUser = new HashMap<Integer, LinkedList<Integer>>();

	/**
	 * CItem[j] = {i|(i, j) \in C}.
	 */
	public HashMap<Integer, LinkedList<Integer>> CItem = new HashMap<Integer, LinkedList<Integer>>();

	/**
	 * User2EventIndexSetMap[i] = {indexOf(i, j) | (i, j) \in C}.
	 */
	public HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
	
	/**
	 * Item2EventIndexSetMap[j] = {indexOf(i, j) | (i, j) \in C}.
	 */
	public HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
	
	/**
	 * EventIndices[k][0]: user index of the k-th event
	 * EventIndices[k][1]: item index of the k-th event
	 */
	// public int[][] EventIndices = null;
	
	/**
	 * Array of feature vectors for all users.
	 */
	public Vector[] Xi = null;
	
	/**
	 * Array of feature vectors for all items.
	 */
	public Vector[] Xj = null;
	
}
