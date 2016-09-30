package ml.recommendation.util;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.TreeMap;

public class Tree {
	
	/**
	 * Geo-tree root: {idx:0, parentIdx:-1}
	 */
	public Node root;
	
	public Tree() {
		root = new Node(0, -1);
	}
	
	public void insertTopCencept(int idx) {
		if (root.children == null)
			root.children = new TreeMap<Integer, Node>();
		TreeMap<Integer, Node> children = root.children;
		
		if (!children.containsKey(idx)) {
			Node node = new Node(idx, root.idx);
			children.put(idx, node);
		}
	}
	
	/**
	 * Insert edge (parent.idx, this.idx).
	 */
	public void insertEdge(Node parent, int pIdx, int idx) {
		if (parent.idx == pIdx) {
			if (parent.children == null)
				parent.children = new TreeMap<Integer, Node>();
			TreeMap<Integer, Node> children = parent.children;
			if (!children.containsKey(idx)) {
				Node node = new Node(idx, root.idx);
				children.put(idx, node);
			}
			return;
		}
		for (Node child : parent.children.values())
			insertEdge(child, pIdx, idx);
	}
	
	/**
	 * Insert a path to the tree.
	 * 
	 * @param path
	 */
	public void insertPath(LinkedList<Integer> path) {
		insertPath(root, path);
	}
	
	/**
	 * Insert a path to a subtree rooted by parent.
	 * 
	 * @param parent
	 * @param path
	 */
	public void insertPath(Node parent, LinkedList<Integer> path) {
		if (path.isEmpty())
			return;
		if (parent.children == null)
			parent.children = new TreeMap<Integer, Node>();
		TreeMap<Integer, Node> children = parent.children;
		
		Node child = null;
		int index = path.pop();
		if (!children.containsKey(index)) {
			child = new Node(index, parent.idx);
			children.put(index, child);
		} else
			child = children.get(index);
		insertPath(child, path);
	}
	
	public void insertLocation(int countryIdx, int stateIdx, int cityIdx) {
		
		if (root.children == null)
			root.children = new TreeMap<Integer, Node>();
		TreeMap<Integer, Node> countries = root.children;
		
		Node country = null;
		if (!countries.containsKey(countryIdx)) {
			country = new Node(countryIdx, root.idx);
			countries.put(countryIdx, country);
		} else
			country = countries.get(countryIdx);
		
		if (country.children == null)
			country.children = new TreeMap<Integer, Node>();
		TreeMap<Integer, Node> states = country.children;
		
		Node state = null;
		if (!states.containsKey(stateIdx)) {
			state = new Node(stateIdx, country.idx);
			states.put(stateIdx, state);
		} else
			state = states.get(stateIdx);
		
		if (state.children == null)
			state.children = new TreeMap<Integer, Node>();
		TreeMap<Integer, Node> cities = state.children;
		
		Node city = null;
		if (!cities.containsKey(cityIdx)) {
			city = new Node(cityIdx, state.idx);
			cities.put(cityIdx, city);
		} else
			city = cities.get(cityIdx);
		
	}
	
	public void print() {
		String indent = "";
		String indentUnit = "    ";
		print(indent, indentUnit);
	}
	
	public void print(String indent, String indentUnit) {
		print(root, indent, indentUnit, 0);
	}
	
	public void print(Node node, String indent, String indentUnit, int level) {
		
		if (node == null)
			return;
		
		// System.out.println();
		System.out.print(indent);
		String type = level == 0 ? "root" : String.format("level %d", level);
		/*switch (level) {
		case 0:
			type = "root";
			break;
		case 1:
			type = "country";
			break;
		case 2:
			type = "state";
			break;
		case 3:
			type = "city";
			break;
		}*/
		System.out.printf("%d (%s)\n", node.idx, type);
		TreeMap<Integer, Node> children = node.children;
		if (children == null)
			return;
		indent = indent + indentUnit;
		level = level + 1;
		for (int idx : children.keySet()) {
			print(children.get(idx), indent, indentUnit, level);
		}
	}
	
	public void save(String filePath) {
		String indent = "";
		String indentUnit = "    ";
		save(filePath, indent, indentUnit);
	}
	
	public void save(String filePath, String indent, String indentUnit) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		save(pw, root, indent, indentUnit, 0);
		pw.close();
	}
	
	public void save(PrintWriter pw, Node node, String indent, String indentUnit, int level) {
		
		if (node == null)
			return;
		
		pw.print(indent);
		// String type = level == 0 ? "root" : String.format("level %d", level);
		/*switch (level) {
		case 0:
			type = "root";
			break;
		case 1:
			type = "country";
			break;
		case 2:
			type = "state";
			break;
		case 3:
			type = "city";
			break;
		}*/
		// pw.printf("%d (%s)\n", node.idx, type);
		pw.printf("%d\n", node.idx);
		TreeMap<Integer, Node> children = node.children;
		if (children == null)
			return;
		indent = indent + indentUnit;
		level = level + 1;
		for (int idx : children.keySet()) {
			save(pw, children.get(idx), indent, indentUnit, level);
		}
	}
	
	public void saveGeoTree(String filePath) {
		String indent = "";
		String indentUnit = "    ";
		saveGeoTree(filePath, indent, indentUnit);
	}
	
	public void saveGeoTree(String filePath, String indent, String indentUnit) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		saveGeoTree(pw, root, indent, indentUnit, 0);
		pw.close();
	}
	
	public void saveGeoTree(PrintWriter pw, Node node, String indent, String indentUnit, int level) {
		
		if (node == null)
			return;
		
		pw.print(indent);
		String type = "";
		switch (level) {
		case 0:
			type = "root";
			break;
		case 1:
			type = "country";
			break;
		case 2:
			type = "state";
			break;
		case 3:
			type = "city";
			break;
		}
		pw.printf("%d (%s)\n", node.idx, type);
		// pw.printf("%d\n", node.idx);
		TreeMap<Integer, Node> children = node.children;
		if (children == null)
			return;
		indent = indent + indentUnit;
		level = level + 1;
		for (int idx : children.keySet()) {
			saveGeoTree(pw, children.get(idx), indent, indentUnit, level);
		}
	}
	
	public void saveWithLevelTags(String filePath) {
		String indent = "";
		String indentUnit = "    ";
		saveWithLevelTags(filePath, indent, indentUnit);
	}
	
	public void saveWithLevelTags(String filePath, String indent, String indentUnit) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		saveWithLevelTags(pw, root, indent, indentUnit, 0);
		pw.close();
	}
	
	public void saveWithLevelTags(PrintWriter pw, Node node, String indent, String indentUnit, int level) {
		
		if (node == null)
			return;
		
		pw.print(indent);
		String type = level == 0 ? "root" : String.format("level %d", level);
		/*switch (level) {
		case 0:
			type = "root";
			break;
		case 1:
			type = "country";
			break;
		case 2:
			type = "state";
			break;
		case 3:
			type = "city";
			break;
		}*/
		pw.printf("%d (%s)\n", node.idx, type);
		// pw.printf("%d\n", node.idx);
		TreeMap<Integer, Node> children = node.children;
		if (children == null)
			return;
		indent = indent + indentUnit;
		level = level + 1;
		for (int idx : children.keySet()) {
			saveWithLevelTags(pw, children.get(idx), indent, indentUnit, level);
		}
	}

}
