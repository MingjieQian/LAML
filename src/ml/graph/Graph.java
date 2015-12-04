package ml.graph;

import java.util.LinkedList;
import java.util.Map.Entry;

import ml.utils.MinPQueue;
import ml.utils.Pair;

public class Graph<K extends Comparable<K>> {

	LinkedList<Vertex<K>> vertices;
	
	public Graph() {
		vertices = new LinkedList<Vertex<K>>();
	}
	
	public void addVertex(Vertex<K> v) {
		vertices.add(v);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Vertex<Double> A = new Vertex<Double>(3.5);
		Vertex<Double> B = new Vertex<Double>(2.5);
		Vertex<Double> C = new Vertex<Double>(5.5);
		Vertex<Double> D = new Vertex<Double>(1.5);
		
		MinPQueue<Vertex<Double>, Double> minQueue = new MinPQueue<Vertex<Double>, Double>();
		minQueue.insert(A);
		minQueue.insert(B);
		minQueue.insert(C);
		minQueue.insert(D);
		System.out.printf("Min key: %f\n", minQueue.delMin().key);
		minQueue.update(A, 12.0);
		System.out.printf("Min key: %f\n", minQueue.delMin().key);
		System.out.printf("Min key: %f\n", minQueue.delMin().key);
		System.out.printf("Min key: %f\n", minQueue.delMin().key);
		
		D.key = 6.5;
		
		Graph<Double> G = new Graph<Double>();
		
		Vertex<Double> a = new Vertex<Double>("a");
		Vertex<Double> b = new Vertex<Double>(Double.class, "b");
		Vertex<Double> c = new Vertex<Double>(Double.class, "c");
		Vertex<Double> d = new Vertex<Double>(Double.class, "d");
		Vertex<Double> e = new Vertex<Double>(Double.class, "e");
		Vertex<Double> f = new Vertex<Double>(Double.class, "f");
		Vertex<Double> g = new Vertex<Double>(Double.class, "g");
		Vertex<Double> h = new Vertex<Double>(Double.class, "h");
		Vertex<Double> i = new Vertex<Double>(Double.class, "i");
		
		a.addToAdjList(b, 4.0);
		a.addToAdjList(h, 8.0);
		
		b.addToAdjList(a, 4.0);
		b.addToAdjList(c, 8.0);
		b.addToAdjList(h, 11.0);
		
		c.addToAdjList(b, 8.0);
		c.addToAdjList(d, 7.0);
		c.addToAdjList(f, 4.0);
		c.addToAdjList(i, 2.0);
		
		d.addToAdjList(c, 7.0);
		d.addToAdjList(e, 9.0);
		d.addToAdjList(f, 14.0);
		
		e.addToAdjList(d, 9.0);
		e.addToAdjList(f, 10.0);

		f.addToAdjList(c, 4.0);
		f.addToAdjList(d, 14.0);
		f.addToAdjList(e, 10.0);
		f.addToAdjList(g, 2.0);
		
		g.addToAdjList(f, 2.0);
		g.addToAdjList(h, 1.0);
		g.addToAdjList(i, 6.0);
		
		h.addToAdjList(a, 8.0);
		h.addToAdjList(b, 11.0);
		h.addToAdjList(g, 1.0);
		h.addToAdjList(i, 7.0);
		
		i.addToAdjList(c, 2.0);
		i.addToAdjList(g, 6.0);
		i.addToAdjList(h, 7.0);

		G.addVertex(a);
		G.addVertex(b);
		G.addVertex(c);
		G.addVertex(d);
		G.addVertex(e);
		G.addVertex(f);
		G.addVertex(g);
		G.addVertex(h);
		G.addVertex(i);
		
		for (Vertex<Double> v : G.vertices) {
			System.out.println(v);
		}
		
		/*HashSet<Vertex<Double>> set = new HashSet<Vertex<Double>>();
		set.add(a);
		set.add(b);
		System.out.print(set.contains(a));
		a.key = new Double(10.0);
		System.out.print(set.contains(a));
		
		HashMap<Vertex<Double>, Integer> map = new HashMap<Vertex<Double>, Integer>();
		map.put(a, 1);
		System.out.print(map.containsKey(a));
		a.key = new Double(11.0);
		System.out.print(map.containsKey(a));*/
		
		LinkedList<Edge<Double>> MST = minimumSpanningTreePrim(G);
		
		for (Edge<Double> edge : MST) {
			System.out.print(edge.edge);
			System.out.printf(": %f\n", edge.weight);
		}
		
		Vertex<Double> s = a;
		LinkedList<Vertex<Double>> S = shortestPathDijkstra(G, s);
		for (Vertex<Double> v : S) {
			if (v == s)
				continue;
			System.out.printf("delta(%s, %s): %s	w(%s->%s): %s \n", s, v, v.key, v.parent, v, v.parent.adjcencyMap.get(v));
		}
	}
	
	@SuppressWarnings("unchecked")
	public static <K extends Comparable<K>> LinkedList<Edge<K>> minimumSpanningTreePrim(Graph<K> G) {
		Vertex<K> r = G.vertices.element();
		for (Vertex<K> v : G.vertices) {
			// v.key = null; // null means positive infinity
			// v.key = (K) new Double(Double.POSITIVE_INFINITY);
			if (v.typeClass == null)
				v.key = null;
			else if (v.typeClass.equals(Double.class))
				v.key = (K) new Double(Double.POSITIVE_INFINITY);
			else if (v.typeClass.equals(Integer.class))
				v.key = (K) new Integer(Integer.MAX_VALUE);
			else if (v.typeClass.equals(Float.class))
				v.key = (K) new Float(Float.POSITIVE_INFINITY);
			// System.out.println(v.key);
			/*System.out.println(v.key);
			if (v.key instanceof Double) {
				v.key = (K) new Double(Double.POSITIVE_INFINITY);
			} else if (v.key instanceof Integer) {
				v.key = (K) new Integer(Integer.MAX_VALUE);
			}*/
			v.parent = null;
		}
		if (r.key instanceof Double) {
			r.key = (K) new Double(0.0);
		} else if (r.key instanceof Integer) {
			r.key = (K) new Integer(0);
		} else if (r.key instanceof Float) {
			r.key = (K) new Float(0);
		} else {
			r.key = null; // null is equivalent to negative infinity
		}
		MinPQueue<Vertex<K>, K> Q = new MinPQueue<Vertex<K>, K>();
		for (Vertex<K> v : G.vertices) {
			Q.insert(v);
		}
		while (!Q.isEmpty()) {
			Vertex<K> u = Q.delMin();
			/*System.out.println(u);
			System.out.println(Q.keyToIndexMap.get(u));
			System.out.println(u.key);
			System.out.println(u.parent);
			System.out.println(u.adjcencyList);*/
			for (Pair<Vertex<K>, K> edge : u.adjcencyList) {
				Vertex<K> v = edge.first;
				K w = edge.second;
				/*System.out.println(v);
				System.out.println(Q.keyToIndexMap.get(v));
				System.out.println(Q.containsKey(v));
				System.out.println(w.compareTo(v.key));*/
				if (Q.containsKey(v) && w.compareTo(v.key) < 0) {
					v.parent = u;
					Q.update(v, w);
				}	
			}
		}
		
		LinkedList<Edge<K>> MST = new LinkedList<Edge<K>>();
		for (Vertex<K> v : G.vertices) {
			if (v != r) {
				MST.add(new Edge<K>(v.parent, v, v.key));
			}
		}
		return MST;
	}
	
	@SuppressWarnings("unchecked")
	public static <K extends Comparable<K>> LinkedList<Vertex<K>> shortestPathDijkstra(Graph<K> G, Vertex<K> s) {
		// v.key is v.d i.e., the upper bound for delta(s, v)
		for (Vertex<K> v : G.vertices) {
			if (v.typeClass == null)
				v.key = null;
			else if (v.typeClass.equals(Double.class))
				v.key = (K) new Double(Double.POSITIVE_INFINITY);
			else if (v.typeClass.equals(Integer.class))
				v.key = (K) new Integer(Integer.MAX_VALUE);
			else if (v.typeClass.equals(Float.class))
				v.key = (K) new Float(Float.POSITIVE_INFINITY);
			v.parent = null;
		}
		if (s.key instanceof Double) {
			s.key = (K) new Double(0.0);
		} else if (s.key instanceof Integer) {
			s.key = (K) new Integer(0);
		} else if (s.key instanceof Float) {
			s.key = (K) new Float(0);
		} else {
			s.key = null; // null is equivalent to negative infinity
		}
		/*if (s.typeClass == Double.class) {
			s.key = (K) new Double(0.0);
		} else if (s.typeClass == Float.class) {
			s.key = (K) new Float(0);
		} else if (s.typeClass == Integer.class) {
			s.key = (K) new Integer(0);
		} else {
			s.key = null;
		}*/
		MinPQueue<Vertex<K>, K> Q = new MinPQueue<Vertex<K>, K>();
		for (Vertex<K> v : G.vertices) {
			Q.insert(v);
		}
		LinkedList<Vertex<K>> S = new LinkedList<Vertex<K>>();
		while (!Q.isEmpty()) {
			Vertex<K> u = Q.delMin();
			S.add(u);
			/*for (Pair<Vertex<K>, K> edge : u.adjcencyList) {
				Vertex<K> v = edge.first;
				K w = edge.second;
				K temp = add(u.key, w);
				if (temp.compareTo(v.key) < 0) {
					v.parent = u;
					Q.update(v, temp);
				}	
			}*/
			for (Entry<Vertex<K>,K> edge : u.adjcencyMap.entrySet()) {
				Vertex<K> v = edge.getKey();
				K w = edge.getValue();
				K temp = add(u.key, w);
				if (temp.compareTo(v.key) < 0) {
					v.parent = u;
					Q.update(v, temp);
				}	
			}
		}
		return S;
	}
	
	@SuppressWarnings("unchecked")
	public static <K> K add(K k1, K k2) {
		K res = null;
		if (k1 instanceof Double) {
			res = (K) new Double(((Double) k1).doubleValue() + ((Double) k2).doubleValue());
		} else if (k1 instanceof Float) {
			res = (K) new Float(((Float) k1).floatValue() + ((Float) k2).floatValue());
		} else if (k1 instanceof Integer) {
			res = (K) new Integer(((Float) k1).intValue() + ((Float) k2).intValue());
		} else {
			res = k2;
		}
		return res;
	}

}
