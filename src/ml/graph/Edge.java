package ml.graph;

import ml.utils.Pair;

public class Edge<K extends Comparable<K>> {
	
	Pair<Vertex<K>, Vertex<K>> edge;
	K weight;

	public Edge(Pair<Vertex<K>, Vertex<K>> edge, K weight) {
		this.edge = edge;
		this.weight = weight;
	}
	
	public Edge(Vertex<K> u, Vertex<K> v, K weight) {
		this.edge = Pair.of(u, v);
		this.weight = weight;
	}
	
}
