package ml.recommendation.util;

import static la.utils.ArrayOperator.sort;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import la.io.DataVectors;
import la.io.InvalidInputDataException;
import la.vector.SparseVector;
import la.vector.Vector;
import la.utils.Pair;

public class Utility {
	
	public static void exit(int status) {
		System.exit(status);
	}
	
	public static <K, V> void saveMap(Map<K, V> map, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		for (Entry<K, V> entry : map.entrySet()) {
			pw.print(entry.getKey());
			pw.print('\t');
			pw.println(entry.getValue());
		}
		pw.close();
	}
	
	public static void addTreeStructuredGroupList(Node parent, ArrayList<ArrayList<Pair<Integer, Integer>>> treeStructuredPairGroupList) {
		// Leaf nodes
		if (parent.children == null) {
			ArrayList<Pair<Integer, Integer>> pairList = nodeEdges(parent);
			if (pairList != null) {
				treeStructuredPairGroupList.add(pairList);
			}
			return;
		}
		TreeMap<Integer, Node> children = parent.children;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			addTreeStructuredGroupList(child, treeStructuredPairGroupList);
		}
		ArrayList<Pair<Integer, Integer>> pairList = nodeEdges(parent);
		if (pairList != null) {
			treeStructuredPairGroupList.add(pairList);
		}	
	}
	
	/**
	 * Compute a pair list composed of descendant edges and parent edge,
	 * which forms a group for the subtree rooted by the parent node.
	 * 
	 * @param parent
	 * 
	 * @return a pair list composed of descendant edges and parent edge
	 */
	@SuppressWarnings("unused")
	private static ArrayList<Pair<Integer, Integer>> nodeEdges0(Node parent) {
		// Root or country nodes
		if (parent.idx == 0 || parent.parentIdx == 0)
			return null;
		// City nodes
		if (parent.children == null) {
			ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
			pairList.add(Pair.of(parent.parentIdx, parent.idx));
			return pairList;
		}
		// State nodes
		ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
		TreeMap<Integer, Node> children = parent.children;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			pairList.addAll(nodeEdges0(child));
		}
		pairList.add(Pair.of(parent.parentIdx, parent.idx));
		return pairList;
	}
	
	/**
	 * Compute a pair list composed of descendant edges and parent edge,
	 * which forms a group for the subtree rooted by the parent node.
	 * 
	 * @param parent
	 * 
	 * @return a pair list composed of descendant edges and parent edge
	 */
	private static ArrayList<Pair<Integer, Integer>> nodeEdges(Node parent) {
		// Root or country nodes
		if (parent.idx == 0 || parent.parentIdx == 0)
			return null;
		// City nodes
		if (parent.children == null) {
			ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
			pairList.add(Pair.of(parent.parentIdx, parent.idx));
			return pairList;
		}
		// State nodes
		ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
		TreeMap<Integer, Node> children = parent.children;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			pairList.addAll(nodeEdges(child));
		}
		pairList.add(0, Pair.of(parent.parentIdx, parent.idx));
		return pairList;
	}

	/**
	 * Traverse and build indices for all the edges (pIdx, idx) on all the nodes
	 * in the subtree rooted by parent node.
	 * 
	 * @param parent
	 * @param pair2IndexMap 
	 * @param index2PairMap 
	 */
	public static void traverseTree(
			Node parent, 
			HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, 
			TreeMap<Integer, Pair<Integer, Integer>> index2PairMap
			) {
		if (parent.children == null) {
			if (parent.idx == 0 || parent.parentIdx == 0)
				return;
			Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
			int index = pair2IndexMap.size();
			pair2IndexMap.put(pair, index);
			index2PairMap.put(index, pair);
			return;
		}
		TreeMap<Integer, Node> children = parent.children;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			traverseTree(child, pair2IndexMap, index2PairMap);
		}
		// Root or country nodes
		if (parent.idx == 0 || parent.parentIdx == 0)
			return;
		// State nodes
		Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
		int index = pair2IndexMap.size();
		pair2IndexMap.put(pair, index);
		index2PairMap.put(index, pair);
	}
	
	/**
	 * Traverse and build indices for all the edges (pIdx, idx) on all the nodes
	 * in the subtree rooted by parent node in post-order.
	 * 
	 * @param parent
	 * @param pair2IndexMap 
	 * @param index2PairMap 
	 */
	public static void postTraverse(
			Node parent, 
			HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, 
			TreeMap<Integer, Pair<Integer, Integer>> index2PairMap
			) {
		if (parent.children == null) {
			if (parent.idx == 0 || parent.parentIdx == 0)
				return;
			Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
			int index = pair2IndexMap.size();
			pair2IndexMap.put(pair, index);
			index2PairMap.put(index, pair);
			return;
		}
		TreeMap<Integer, Node> children = parent.children;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			postTraverse(child, pair2IndexMap, index2PairMap);
		}
		// Root or country nodes
		if (parent.idx == 0 || parent.parentIdx == 0)
			return;
		// State nodes
		Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
		int index = pair2IndexMap.size();
		pair2IndexMap.put(pair, index);
		index2PairMap.put(index, pair);
	}
	
	/**
	 * Traverse and build indices for all the edges (pIdx, idx) on all the nodes
	 * in the subtree rooted by parent node in pre-order.
	 * 
	 * @param parent
	 * @param pair2IndexMap 
	 * @param index2PairMap 
	 */
	public static void preTraverse(
			Node parent, 
			HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, 
			TreeMap<Integer, Pair<Integer, Integer>> index2PairMap
			) {
		
		if (parent.parentIdx != 0 && parent.idx != 0) {
			Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
			int index = pair2IndexMap.size();
			pair2IndexMap.put(pair, index);
			index2PairMap.put(index, pair);
		}
		
		TreeMap<Integer, Node> children = parent.children;
		if (children == null)
			return;
		for (int idx : children.keySet()) {
			Node child = children.get(idx);
			preTraverse(child, pair2IndexMap, index2PairMap);
		}
		
	}

	public static void buildTreeStructuredIndexGroupList(
			ArrayList<ArrayList<Pair<Integer, Integer>>> treeStructuredPairGroupList,
			HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, 
			ArrayList<ArrayList<Integer>> treeStructuredIndexGroupList
			) {
		for (ArrayList<Pair<Integer, Integer>> pairList : treeStructuredPairGroupList) {
			ArrayList<Integer> indexList = new ArrayList<Integer>();
			for (Pair<Integer, Integer> pair : pairList) {
				int index = pair2IndexMap.get(pair);
				indexList.add(index);
			}
			treeStructuredIndexGroupList.add(indexList);
		}
	}

	public static <T> void saveTreeStructuredGroupList(
			ArrayList<ArrayList<T>> treeStructuredGroupList,
			String filePath) {
		
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		for (ArrayList<T> list : treeStructuredGroupList) {
			StringBuilder sb = new StringBuilder(10);
			for (T element : list) {
				sb.append(element);
				sb.append("\t");
			}
			pw.println(sb.toString().trim());
		}
		pw.close();
		
	}
	
	
	/**
	 * Build TestUser2EventIndexSetMap and TestUserIndices.
	 * 
	 * @param eventFilePath
	 * 
	 * @param TestUser2EventIndexSetMap TestUser2EventIndexSetMap[i] = {indexOf(i, j) | (i, j) \in C}
	 * 
	 * @return TestUserIndices TestUserIndices[k] is the user index for the k-th event
	 */
	public static int[] loadTestUserEventRelation(
			String eventFilePath,
			HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap
			) {
		if (!new File(eventFilePath).exists()) {
			System.err.println(String.format("Event file %s doesn't exist.\n", eventFilePath));
			exit(1);
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(eventFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		String line = "";
		List<Double> YijList = new LinkedList<Double>();
		// List<Vector> XijVectorList = new LinkedList<Vector>();
		List<double[]> XijList = new LinkedList<double[]>();
		List<Integer> userIdxList = new LinkedList<Integer>();
		List<Integer> itemIdxList = new LinkedList<Integer>();
		String[] container = null;
		double label = 0;
		int userIdx = -1;
		int itemIdx = -1;
		double gmp = 0;
		double freshness = 0;
		int eventIdx = -1;
		int maxUserIdx = -1;
		int maxItemIdx = -1;
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = line.split("\t");
				label = Double.parseDouble(container[0]);
				userIdx = Integer.parseInt(container[1]);
				itemIdx = Integer.parseInt(container[2]);
				// Modified on Oct. 16th, 2014
				// {
				if (maxUserIdx < userIdx)
					maxUserIdx = userIdx;
				if (maxItemIdx < itemIdx)
					maxItemIdx = itemIdx;
				// }
				gmp = Double.parseDouble(container[3]);
				freshness = Double.parseDouble(container[4]);
				YijList.add(label);
				userIdxList.add(userIdx);
				itemIdxList.add(itemIdx);
				// XijVectorList.add(new DenseVector(new double[]{gmp, freshness}));
				XijList.add(new double[]{gmp, freshness});
				eventIdx += 1;
				if (TestUser2EventIndexSetMap.containsKey(userIdx)) {
					TestUser2EventIndexSetMap.get(userIdx).add(eventIdx);
				} else {
					LinkedList<Integer> eventSet = new LinkedList<Integer>();
					eventSet.add(eventIdx);
					TestUser2EventIndexSetMap.put(userIdx, eventSet);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int eventCnt = YijList.size();
		int[] TestUserIndices = new int[eventCnt];
		int cnt = 0;
		/*cnt = 0;
		for (double element : YijList) {
			Yij[cnt++] = element;
		}*/
		cnt = 0;
		for (int element : userIdxList) {
			TestUserIndices[cnt++] = element;
		}
		return TestUserIndices;
	}
	
	public static Data loadData(
			String appDirPath,
			String eventFileName, 
			String userFileName, 
			String itemFileName, 
			int[] featureSizes
			) {
		
		Data data = new Data();
		
		int Pu = featureSizes[0];
		int Pv = featureSizes[1];
		int Pe = featureSizes[2];
		
		data.Pu = Pu;
		data.Pv = Pv;
		data.Pe = Pe;
		
		// Load events
		System.out.println("Loading events...");

		/**
		 * Number of users.
		 */
		int M = 0;

		/**
		 * Number of items.
		 */
		int N = 0;

		/**
		 * Number of events.
		 */
		int T = 0;

		/**
		 * Array of labels for all events.
		 */
		double[] Yij = null;
		/**
		 * Array of feature vectors for all events.
		 */
		double[][] Xij = null;

		/**
		 * UserIndices[k] is the user index for the k-th event.
		 */
		int[] UserIndices = null;

		/**
		 * ItemIndices[k] is the item index for the k-th event.
		 */
		int[] ItemIndices = null;

		/**
		 * CUser[i] = {j|(i, j) \in C}.
		 */
		HashMap<Integer, LinkedList<Integer>> CUser = new HashMap<Integer, LinkedList<Integer>>();

		/**
		 * CItem[j] = {i|(i, j) \in C}.
		 */
		HashMap<Integer, LinkedList<Integer>> CItem = new HashMap<Integer, LinkedList<Integer>>();

		/**
		 * User2EventIndexSetMap[i] = {indexOf(i, j) | (i, j) \in C}.
		 */
		HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();

		/**
		 * Item2EventIndexSetMap[j] = {indexOf(i, j) | (i, j) \in C}.
		 */
		HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();

		String eventFilePath = appDirPath + File.separator + eventFileName;
		if (!new File(eventFilePath).exists()) {
			System.err.println(String.format("Event file %s doesn't exist.\n", eventFilePath));
			exit(1);
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(eventFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		String line = "";
		List<Double> YijList = new LinkedList<Double>();
		// List<Vector> XijVectorList = new LinkedList<Vector>();
		List<double[]> XijList = new LinkedList<double[]>();
		List<Integer> userIdxList = new LinkedList<Integer>();
		List<Integer> itemIdxList = new LinkedList<Integer>();
		String[] container = null;
		double label = 0;
		int userIdx = -1;
		int itemIdx = -1;
		double gmp = 0;
		double freshness = 0;
		int eventIdx = -1;
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = line.split("\t");
				label = Double.parseDouble(container[0]);
				userIdx = Integer.parseInt(container[1]);
				itemIdx = Integer.parseInt(container[2]);
				gmp = Double.parseDouble(container[3]);
				freshness = Double.parseDouble(container[4]);
				YijList.add(label);
				userIdxList.add(userIdx);
				itemIdxList.add(itemIdx);
				// XijVectorList.add(new DenseVector(new double[]{gmp, freshness}));
				XijList.add(new double[]{gmp, freshness});
				if (CUser.containsKey(userIdx)) {
					CUser.get(userIdx).add(itemIdx);
				} else {
					LinkedList<Integer> itemSet = new LinkedList<Integer>();
					itemSet.add(itemIdx);
					CUser.put(userIdx, itemSet);
				}
				if (CItem.containsKey(itemIdx)) {
					CItem.get(itemIdx).add(userIdx);
				} else {
					LinkedList<Integer> userSet = new LinkedList<Integer>();
					userSet.add(userIdx);
					CItem.put(itemIdx, userSet);
				}
				eventIdx += 1;
				if (User2EventIndexSetMap.containsKey(userIdx)) {
					User2EventIndexSetMap.get(userIdx).add(eventIdx);
				} else {
					LinkedList<Integer> eventSet = new LinkedList<Integer>();
					eventSet.add(eventIdx);
					User2EventIndexSetMap.put(userIdx, eventSet);
				}
				if (Item2EventIndexSetMap.containsKey(itemIdx)) {
					Item2EventIndexSetMap.get(itemIdx).add(eventIdx);
				} else {
					LinkedList<Integer> eventSet = new LinkedList<Integer>();
					eventSet.add(eventIdx);
					Item2EventIndexSetMap.put(itemIdx, eventSet);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int eventCnt = YijList.size();
		Yij = new double[eventCnt];
		// XijVectors = new Vector[eventCnt];
		Xij = new double[eventCnt][];
		UserIndices = new int[eventCnt];
		ItemIndices = new int[eventCnt];
		int cnt = 0;
		cnt = 0;
		for (double element : YijList) {
			Yij[cnt++] = element;
		}
		cnt = 0;
		for (double[] element : XijList) {
			// XijVectors[cnt++] = element;
			Xij[cnt++] = element;
		}
		cnt = 0;
		for (int element : userIdxList) {
			UserIndices[cnt++] = element;
		}
		cnt = 0;
		for (int element : itemIdxList) {
			ItemIndices[cnt++] = element;
		}
		
		M = CUser.size();
		N = CItem.size();
		T = eventCnt;
		
		String filePath = "";
		DataVectors dataVectors = null;
		
		// Load users
		System.out.println("Loading users...");
		Vector[] Xi = null;
		filePath = appDirPath + File.separator + userFileName;
		DataVectors.IdxStart = 0;
		try {
			dataVectors = la.io.DataVectors.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		Xi = dataVectors.Vs;

		// Load items
		System.out.println("Loading items...");
		Vector[] Xj = null;
		filePath = appDirPath + File.separator + itemFileName;
		DataVectors.IdxStart = 0;
		try {
			dataVectors = la.io.DataVectors.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		Xj = dataVectors.Vs;
		for (int j = 0; j < N; j++) {
			((SparseVector) Xj[j]).setDim(Pv);
		}

		// Feed data
		
		data.M = M;
		data.N = N;
		data.T = T;
		data.Yij = Yij;
		data.Xij = Xij;
		data.UserIndices = UserIndices;
		data.ItemIndices = ItemIndices;
		data.CUser = CUser;
		data.CItem = CItem;
		data.User2EventIndexSetMap = User2EventIndexSetMap;
		data.Item2EventIndexSetMap = Item2EventIndexSetMap;
		data.Xi = Xi;
		data.Xj = Xj;
		
		return data;
		
	}
	
	public static int[] getFeatureSize(String featureSizeFilePath) {
		BufferedReader br = null;
		String line = "";
		String[] container = null;
		String lastLine = "";
		
		/**
		 * User feature size.
		 */
		int Pu = 0;
		
		/**
		 * Item feature size.
		 */
		int Pv = 0;
		
		/**
		 * Event feature size.
		 */
		int Pe = 0;
		
		if (!new File(featureSizeFilePath).exists()) {
			System.err.println(String.format("File %s doesn't exist.\n", featureSizeFilePath));
			exit(1);
		}
		try {
			br = new BufferedReader(new FileReader(featureSizeFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = lastLine.split("\t");
				if (container[0].equals("User")) {
					Pu = Integer.parseInt(container[1]);
				} else if (container[0].equals("Item")) {
					Pv = Integer.parseInt(container[1]);
				} else if (container[0].equals("Event")) {
					Pe = Integer.parseInt(container[1]);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		int[] featureSizes = new int[3];
		featureSizes[0] = Pu;
		featureSizes[1] = Pv;
		featureSizes[2] = Pe;
		return featureSizes;
	}
	
	public static int[] getFeatureSize(
			String appDirPath,
			String eventFileName, 
			String userFeatureMapFileName, 
			String itemFeatureMapFileName
			) {
		
		String filePath = "";
		BufferedReader br = null;
		String line = "";
		String[] container = null;
		String lastLine = "";
		
		/**
		 * User feature size.
		 */
		int Pu = 0;
		
		/**
		 * Item feature size.
		 */
		int Pv = 0;
		
		/**
		 * Event feature size.
		 */
		int Pe = 0;
		
		filePath = appDirPath + File.separator + userFeatureMapFileName;
		if (!new File(filePath).exists()) {
			System.err.println(String.format("File %s doesn't exist.\n", filePath));
			exit(1);
		}
		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				lastLine = line;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		container = lastLine.split("\t");
		Pu = Integer.parseInt(container[0]) + 1;
		
		filePath = appDirPath + File.separator + itemFeatureMapFileName;
		if (!new File(filePath).exists()) {
			System.err.println(String.format("File %s doesn't exist.\n", filePath));
			exit(1);
		}
		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				lastLine = line;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		container = lastLine.split("\t");
		Pv = Integer.parseInt(container[0]) + 1;
		
		filePath = appDirPath + File.separator + eventFileName;
		if (!new File(filePath).exists()) {
			System.err.println(String.format("File %s doesn't exist.\n", filePath));
			exit(1);
		}
		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				break;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		container = line.split("\t");
		Pe = container.length - 3;
		
		int[] featureSizes = new int[3];
		featureSizes[0] = Pu;
		featureSizes[1] = Pv;
		featureSizes[2] = Pe;
		return featureSizes;
		
	}

	public static void saveString(String filePath, String content) {
		PrintWriter pw = null;
		boolean autoFlush = true;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), autoFlush);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		pw.print(content);
		pw.close();
	}
	
	public static int[] getNumSelfEdge(String GEOIndexFilePath,
			String YCTIndexFilePath) {
		BufferedReader br = null;
		String line = "";
		String lastLine = "";
		int numSelfEdgeUser = 0;
		int numSelfEdgeItem = 0;
		try {
			br = new BufferedReader(new FileReader(GEOIndexFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				lastLine = line;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		numSelfEdgeUser = Integer.parseInt(lastLine.trim()) + 1;
		try {
			br = new BufferedReader(new FileReader(YCTIndexFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				lastLine = line;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		numSelfEdgeItem = Integer.parseInt(lastLine.trim()) + 1;
		return new int[]{numSelfEdgeUser, numSelfEdgeItem};
	}
	
	public static String executeCommand(String command) {
		 
		StringBuffer output = new StringBuffer();
 
		Process p;
		try {
			p = Runtime.getRuntime().exec(command);
			
			BufferedReader reader = 
                            new BufferedReader(new InputStreamReader(p.getInputStream()));
 
			String line = "";			
			while ((line = reader.readLine())!= null) {
				output.append(line + "\n");
			}
			
			// p.waitFor();
 
		} catch (Exception e) {
			e.printStackTrace();
		}
 
		return output.toString();
 
	}
	
	public static void saveMeasures(String appDirPath, String fileName, double[] measures) {
		String filePath = appDirPath + File.separator + fileName;
		PrintWriter pw = null;
		boolean autoFlush = true;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), autoFlush);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		double RMSE = measures[0];
		double MAP = measures[1];
		double MRR = measures[2];
		double MP10 = measures[3];
		pw.printf("RMSE: %.8g\n", RMSE);
		pw.printf("MAP: %.8g\n", MAP);
		pw.printf("MRR: %.8g\n", MRR);
		pw.printf("MP@10: %.8g\n", MP10);
		pw.close();
	}
	
	public static void loadMap(TreeMap<Integer, Integer> map, String filePath) {
		BufferedReader br = null;
		String line = "";
		String[] container = null;
		try {
			br = new BufferedReader(new FileReader(filePath));
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = line.split("\t");
				map.put(Integer.valueOf(container[0]), Integer.valueOf(container[1]));
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static double[] predict(
			Data testData, 
			double[] Yij_pred, 
			TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap,
			TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap
			) {
		double[] Yij = testData.Yij;
		// double[][] Xij = testData.Xij;
		// int[] UserIndices = testData.UserIndices;
		// int[] ItemIndices = testData.ItemIndices;
		// HashMap<Integer, LinkedList<Integer>> CUser = testData.CUser;
		// HashMap<Integer, LinkedList<Integer>> CItem = testData.CItem;
		HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
		// HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = testData.Item2EventIndexSetMap;
		// Vector[] Xi = testData.Xi;
		// Vector[] Xj = testData.Xj;
		if (!drawMAPCurve)
			System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
		
		int T = testData.T;
		int M = testData.M;
		// int N = Xj.length;
		
		// Compute RMSE
		
		double RMSE = 0;
		for (int e = 0; e < T; e++) {
			double y = Yij[e];
			double y_pred = Yij_pred[e];
			double error = y - y_pred;
			RMSE += error * error;
		}
		RMSE /= T; // MSE
		RMSE = Math.sqrt(RMSE); // RMSE
		
		// Compute MAP
		
		double MAP = 0;
		double MRR = 0;
		double MP10 = 0;
		for (int i = 0; i < M; i++) {
			/*
			 *  Ideally we should rank all items, but for evaluation,
			 *  like in TREC tasks, we can use a pool set, that said,
			 *  the set of items with which a user has interaction in
			 *  the test set.
			 */
			/*double[] scoresOfItems4UserI = new double[N];
			double[] u_i = U_pred[i];
			double a_i = u_i[K];
			for (int j = 0; j < N; j++) {
				double[] v_j = V_pred[j];
				double b_j = v_j[K];
				scoresOfItems4UserI[j] = innerProduct(W, Xij[e]) - a_i - b_j - innerProduct(u_i, v_j, 0, K);
			}*/
			LinkedList<Integer> eventIdxList = User2EventIndexSetMap.get(i);
			double[] scoresOfItems4UserI = new double[eventIdxList.size()];
			double[] groundTruth4UserI = new double[eventIdxList.size()];
			int k = 0;
			for (int eventIdx : eventIdxList) {
				scoresOfItems4UserI[k] = Yij_pred[eventIdx];
				groundTruth4UserI[k++] = Yij[eventIdx];
			}
			// Sort ranking scores of items for user i
			int[] eventListPosition4RankAt = sort(scoresOfItems4UserI, "descend");
			double AP_i = 0;
			double RR_i = 0;
			double P10_i = 0;
			int numRelItems = 0;
			for (k = 0; k < eventIdxList.size(); k++) {
				int eventListPos = eventListPosition4RankAt[k];
				if (groundTruth4UserI[eventListPos] == 1.0) { // Relevant item
					numRelItems++;
					AP_i += numRelItems / (k + 1.0);
					if (numRelItems == 1) {
						RR_i += 1 / (k + 1.0);
					}
					if (k < 10) {
						P10_i += 0.1;
					}
				} else { // Irrelevant item
					
				}
			}
			AP_i /= numRelItems;
			MAP += AP_i;
			MRR += RR_i;
			MP10 += P10_i;
		}
		MAP /= M;
		MRR /= M;
		MP10 /= M;

		if (!drawMAPCurve) {
			System.out.printf("RMSE: %.6g\n", RMSE);
			System.out.printf("MAP: %.6g\n", MAP);
			System.out.printf("MRR: %.6g\n", MRR);
			System.out.printf("MP@10: %6g\n", MP10);
		}

		double[] res = new double[4];
		res[0] = RMSE;
		res[1] = MAP;
		res[2] = MRR;
		res[3] = MP10;
		
		return res;
		
	}
	
	public static double[] predict(
			int[] Yij,
			double[] Yij_pred, 
			HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap
			) {
		// double[] Yij = testData.Yij;
		// double[][] Xij = testData.Xij;
		// int[] UserIndices = testData.UserIndices;
		// int[] ItemIndices = testData.ItemIndices;
		// HashMap<Integer, LinkedList<Integer>> CUser = testData.CUser;
		// HashMap<Integer, LinkedList<Integer>> CItem = testData.CItem;
		// HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
		// HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = testData.Item2EventIndexSetMap;
		// Vector[] Xi = testData.Xi;
		// Vector[] Xj = testData.Xj;
		if (!drawMAPCurve)
			System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
		
		// int T = testData.T;
		int T = Yij.length;
		// int M = testData.M;
		/*
		 * Note that the keys of TestUser2EventIndexSetMap are
		 * 0, 1, 2, ..., M - 1
		 */
		int M = TestUser2EventIndexSetMap.size();
		// int N = Xj.length;
		
		// Compute RMSE
		
		double RMSE = 0;
		for (int e = 0; e < T; e++) {
			double y = Yij[e];
			double y_pred = Yij_pred[e];
			double error = y - y_pred;
			RMSE += error * error;
		}
		RMSE /= T; // MSE
		RMSE = Math.sqrt(RMSE); // RMSE
		
		// Compute MAP
		
		double MAP = 0;
		double MRR = 0;
		double MP10 = 0;
		for (int i = 0; i < M; i++) {
			/*
			 *  Ideally we should rank all items, but for evaluation,
			 *  like in TREC tasks, we can use a pool set, that said,
			 *  the set of items with which a user has interaction in
			 *  the test set.
			 */
			/*double[] scoresOfItems4UserI = new double[N];
			double[] u_i = U_pred[i];
			double a_i = u_i[K];
			for (int j = 0; j < N; j++) {
				double[] v_j = V_pred[j];
				double b_j = v_j[K];
				scoresOfItems4UserI[j] = innerProduct(W, Xij[e]) - a_i - b_j - innerProduct(u_i, v_j, 0, K);
			}*/
			LinkedList<Integer> eventIdxList = TestUser2EventIndexSetMap.get(i);
			double[] scoresOfItems4UserI = new double[eventIdxList.size()];
			double[] groundTruth4UserI = new double[eventIdxList.size()];
			int k = 0;
			for (int eventIdx : eventIdxList) {
				scoresOfItems4UserI[k] = Yij_pred[eventIdx];
				groundTruth4UserI[k++] = Yij[eventIdx];
			}
			// Sort ranking scores of items for user i
			int[] eventListPosition4RankAt = sort(scoresOfItems4UserI, "descend");
			double AP_i = 0;
			double RR_i = 0;
			double P10_i = 0;
			int numRelItems = 0;
			for (k = 0; k < eventIdxList.size(); k++) {
				int eventListPos = eventListPosition4RankAt[k];
				if (groundTruth4UserI[eventListPos] == 1.0) { // Relevant item
					numRelItems++;
					AP_i += numRelItems / (k + 1.0);
					if (numRelItems == 1) {
						RR_i += 1 / (k + 1.0);
					}
					if (k < 10) {
						P10_i += 0.1;
					}
				} else { // Irrelevant item
					
				}
			}
			AP_i /= numRelItems;
			MAP += AP_i;
			MRR += RR_i;
			MP10 += P10_i;
		}
		MAP /= M;
		MRR /= M;
		MP10 /= M;

		if (!drawMAPCurve) {
			System.out.printf("RMSE: %.6g\n", RMSE);
			System.out.printf("MAP: %.6g\n", MAP);
			System.out.printf("MRR: %.6g\n", MRR);
			System.out.printf("MP@10: %6g\n", MP10);
		}

		double[] res = new double[4];
		res[0] = RMSE;
		res[1] = MAP;
		res[2] = MRR;
		res[3] = MP10;
		
		return res;
		
	}

	public static double[] predictColdStart(
			Data testData, 
			double[] Yij_pred, 
			TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap,
			TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap
			) {
		
		double[] Yij = testData.Yij;
		// double[][] Xij = testData.Xij;
		int[] UserIndices = testData.UserIndices;
		// int[] ItemIndices = testData.ItemIndices;
		// HashMap<Integer, LinkedList<Integer>> CUser = testData.CUser;
		// HashMap<Integer, LinkedList<Integer>> CItem = testData.CItem;
		HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
		// HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = testData.Item2EventIndexSetMap;
		// Vector[] Xi = testData.Xi;
		// Vector[] Xj = testData.Xj;
		int M = testData.M;
		// int N = testData.N;
		int T = testData.T;
		
		if (!drawMAPCurve) {
			System.out.println("\nCold start setting:");
			System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
		}
		
		// Compute RMSE
		int numColdStart = 0;
		
		double RMSE = 0;
		for (int e = 0; e < T; e++) {
			double y = Yij[e];
			double y_pred = Yij_pred[e];
			int i = UserIndices[e];
			// int j = ItemIndices[e];
			int trainIdx = TestIdx2TrainIdxUserMap.get(i);
			if (trainIdx != -1) // Old user
				continue;
			numColdStart++;
			double error = y - y_pred;
			RMSE += error * error;
		}
		RMSE /= numColdStart; // MSE
		RMSE = Math.sqrt(RMSE); // RMSE
		
		// Compute MAP
		int numNewUsers = 0;
		double MAP = 0;
		double MRR = 0;
		double MP10 = 0;
		for (int i = 0; i < M; i++) {
			int trainIdx = TestIdx2TrainIdxUserMap.get(i);
			if (trainIdx != -1) // Old user
				continue;
			numNewUsers++;
			/*
			 *  Ideally we should rank all items, but for evaluation,
			 *  like in TREC tasks, we can use a pool set, that said,
			 *  the set of items with which a user has interaction in
			 *  the test set.
			 */
			/*double[] scoresOfItems4UserI = new double[N];
			double[] u_i = U_pred[i];
			double a_i = u_i[K];
			for (int j = 0; j < N; j++) {
				double[] v_j = V_pred[j];
				double b_j = v_j[K];
				scoresOfItems4UserI[j] = innerProduct(W, Xij[e]) - a_i - b_j - innerProduct(u_i, v_j, 0, K);
			}*/
			
			LinkedList<Integer> eventIdxList = User2EventIndexSetMap.get(i);
	
			/*
			 * We need to filtered out all old items in the eventIdxList. 
			 */
			/*LinkedList<Integer> eventIdxList = new LinkedList<Integer>();
			for (int eventIdx : eventIdxList_ori) {
				int j = ItemIndices[eventIdx];
				String itemID = TestIndex2ItemIDMap.get(j);
				if (!TrainItemID2IndexMap.containsKey(itemID)) // The new user clicked a new item in this event
					eventIdxList.add(eventIdx);
			}*/
			
			double[] scoresOfItems4UserI = new double[eventIdxList.size()];
			double[] groundTruth4UserI = new double[eventIdxList.size()];
			int k = 0;
			for (int eventIdx : eventIdxList) {
				scoresOfItems4UserI[k] = Yij_pred[eventIdx];
				groundTruth4UserI[k++] = Yij[eventIdx];
			}
			// Sort ranking scores of items for user i
			int[] eventListPosition4RankAt = sort(scoresOfItems4UserI, "descend");
			double AP_i = 0;
			double RR_i = 0;
			double P10_i = 0;
			int numRelItems = 0;
			for (k = 0; k < eventIdxList.size(); k++) {
				int eventListPos = eventListPosition4RankAt[k];
				if (groundTruth4UserI[eventListPos] == 1.0) { // Relevant item
					numRelItems++;
					AP_i += numRelItems / (k + 1.0);
					if (numRelItems == 1) {
						RR_i += 1 / (k + 1.0);
					}
					if (k < 10) {
						P10_i += 0.1;
					}
				} else { // Irrelevant item
					
				}
			}
			AP_i /= numRelItems;
			MAP += AP_i;
			MRR += RR_i;
			MP10 += P10_i;
		}
		MAP /= numNewUsers;
		MRR /= numNewUsers;
		MP10 /= numNewUsers;
		
		if (!drawMAPCurve) {
			System.out.printf("RMSE: %.6g\n", RMSE);
			System.out.printf("MAP: %.6g\n", MAP);
			System.out.printf("MRR: %.6g\n", MRR);
			System.out.printf("MP@10: %6g\n", MP10);
		}
		
		double[] res = new double[4];
		res[0] = RMSE;
		res[1] = MAP;
		res[2] = MRR;
		res[3] = MP10;
		
		return res;
		
	}
	
	public static double[] predictColdStart(
			int[] Yij, 
			double[] Yij_pred,
			int[] TestUserIndices,
			HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap,
			TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap,
			TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap
			) {
		
		// double[] Yij = testData.Yij;
		// double[][] Xij = testData.Xij;
		// int[] UserIndices = testData.UserIndices;
		// int[] ItemIndices = testData.ItemIndices;
		// HashMap<Integer, LinkedList<Integer>> CUser = testData.CUser;
		// HashMap<Integer, LinkedList<Integer>> CItem = testData.CItem;
		// HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
		// HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = testData.Item2EventIndexSetMap;
		// Vector[] Xi = testData.Xi;
		// Vector[] Xj = testData.Xj;
		// int M = testData.M;
		int M = TestUser2EventIndexSetMap.size();
		// int N = testData.N;
		// int T = testData.T;
		int T = Yij.length;
		
		if (!drawMAPCurve) {
			System.out.println("\nCold start setting:");
			System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
		}
		
		// Compute RMSE
		int numColdStart = 0;
		
		double RMSE = 0;
		for (int e = 0; e < T; e++) {
			double y = Yij[e];
			double y_pred = Yij_pred[e];
			int i = TestUserIndices[e];
			// int j = ItemIndices[e];
			int trainIdx = TestIdx2TrainIdxUserMap.get(i);
			if (trainIdx != -1) // Old user
				continue;
			numColdStart++;
			double error = y - y_pred;
			RMSE += error * error;
		}
		RMSE /= numColdStart; // MSE
		RMSE = Math.sqrt(RMSE); // RMSE
		
		// Compute MAP
		int numNewUsers = 0;
		double MAP = 0;
		double MRR = 0;
		double MP10 = 0;
		for (int i = 0; i < M; i++) {
			int trainIdx = TestIdx2TrainIdxUserMap.get(i);
			if (trainIdx != -1) // Old user
				continue;
			numNewUsers++;
			/*
			 *  Ideally we should rank all items, but for evaluation,
			 *  like in TREC tasks, we can use a pool set, that said,
			 *  the set of items with which a user has interaction in
			 *  the test set.
			 */
			/*double[] scoresOfItems4UserI = new double[N];
			double[] u_i = U_pred[i];
			double a_i = u_i[K];
			for (int j = 0; j < N; j++) {
				double[] v_j = V_pred[j];
				double b_j = v_j[K];
				scoresOfItems4UserI[j] = innerProduct(W, Xij[e]) - a_i - b_j - innerProduct(u_i, v_j, 0, K);
			}*/
			
			LinkedList<Integer> eventIdxList = TestUser2EventIndexSetMap.get(i);
	
			/*
			 * We need to filtered out all old items in the eventIdxList. 
			 */
			/*LinkedList<Integer> eventIdxList = new LinkedList<Integer>();
			for (int eventIdx : eventIdxList_ori) {
				int j = ItemIndices[eventIdx];
				String itemID = TestIndex2ItemIDMap.get(j);
				if (!TrainItemID2IndexMap.containsKey(itemID)) // The new user clicked a new item in this event
					eventIdxList.add(eventIdx);
			}*/
			
			double[] scoresOfItems4UserI = new double[eventIdxList.size()];
			double[] groundTruth4UserI = new double[eventIdxList.size()];
			int k = 0;
			for (int eventIdx : eventIdxList) {
				scoresOfItems4UserI[k] = Yij_pred[eventIdx];
				groundTruth4UserI[k++] = Yij[eventIdx];
			}
			// Sort ranking scores of items for user i
			int[] eventListPosition4RankAt = sort(scoresOfItems4UserI, "descend");
			double AP_i = 0;
			double RR_i = 0;
			double P10_i = 0;
			int numRelItems = 0;
			for (k = 0; k < eventIdxList.size(); k++) {
				int eventListPos = eventListPosition4RankAt[k];
				if (groundTruth4UserI[eventListPos] == 1.0) { // Relevant item
					numRelItems++;
					AP_i += numRelItems / (k + 1.0);
					if (numRelItems == 1) {
						RR_i += 1 / (k + 1.0);
					}
					if (k < 10) {
						P10_i += 0.1;
					}
				} else { // Irrelevant item
					
				}
			}
			AP_i /= numRelItems;
			MAP += AP_i;
			MRR += RR_i;
			MP10 += P10_i;
		}
		MAP /= numNewUsers;
		MRR /= numNewUsers;
		MP10 /= numNewUsers;
		
		if (!drawMAPCurve) {
			System.out.printf("RMSE: %.6g\n", RMSE);
			System.out.printf("MAP: %.6g\n", MAP);
			System.out.printf("MRR: %.6g\n", MRR);
			System.out.printf("MP@10: %6g\n", MP10);
		}
		
		double[] res = new double[4];
		res[0] = RMSE;
		res[1] = MAP;
		res[2] = MRR;
		res[3] = MP10;
		
		return res;
		
	}
	
	public static boolean drawMAPCurve = false;
	
}
