package org.laml.ml.recommendation.util;

import java.util.TreeMap;

public class Node {
	public int idx;
	public int parentIdx;
	public TreeMap<Integer, Node> children;
	
	public Node() {
		idx = 0;
		parentIdx = 0;
		children = null;
	}
	
	public Node(int idx, int parentIdx) {
		this.idx = idx;
		this.parentIdx = parentIdx;
		this.children = null; 
	}
}
