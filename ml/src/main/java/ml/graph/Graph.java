package ml.graph;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import la.utils.PriorityQueue;

public class Graph<K> {

    public LinkedList<Vertex<K>> vertices;
    
    public Graph() {
        vertices = new LinkedList<Vertex<K>>();
    }
    
    public void addVertex(Vertex<K> v) {
        vertices.add(v);
    }
    
    /**
	 * Deep copy of this graph.
	 */
	public Graph<K> clone() {
		HashMap<Vertex<K>, Vertex<K>> map = new HashMap<Vertex<K>, Vertex<K>>();
		Graph<K> res = new Graph<K>();
		for (Vertex<K> u : vertices) {
			Vertex<K> u_copy = null;
			if (map.containsKey(u)) {
				u_copy = map.get(u);
			} else {
				u_copy = new Vertex<K>(u.key, u.name);
			}
			for (Entry<Vertex<K>, Double> entry : u.adjacencyMap.entrySet()) {
				Vertex<K> v = entry.getKey();
				double w = entry.getValue();
				Vertex<K> v_copy = null;
				if (map.containsKey(v)) {
					v_copy = map.get(v);
				} else {
					v_copy = new Vertex<K>(v.key, v.name);
				}
				u_copy.addToAdjList(v_copy, w);
			}
			res.addVertex(u_copy);
		}
		return res;
	}
    
    /**
     * @param args
     */
    public static void main(String[] args) {
        
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
        
        LinkedList<Edge<Double>> MST = minimumSpanningTreePrim(G);
        
        System.out.println("Minimum spanning tree:");
        for (Edge<Double> edge : MST) {
            System.out.println(edge);
            // System.out.printf(": %f\n", edge.weight);
        }
        
        Vertex<Double> s = a;
        Map<Vertex<Double>, Double> S = shortestPathDijkstra(G, s);
        for (Vertex<Double> v : S.keySet()) {
            if (v == s)
                continue;
            System.out.printf("delta(%s, %s): %s    w(%s->%s): %s \n", s, v, S.get(v), v.parent, v, v.parent.adjacencyMap.get(v));
        }
        
        G = new Graph<Double>();
        
        a = new Vertex<Double>(Double.class, "a");
        b = new Vertex<Double>(Double.class, "b");
        c = new Vertex<Double>(Double.class, "c");
        d = new Vertex<Double>(Double.class, "d");
        e = new Vertex<Double>(Double.class, "e");
        f = new Vertex<Double>(Double.class, "f");
        g = new Vertex<Double>(Double.class, "g");
        h = new Vertex<Double>(Double.class, "h");
        i = new Vertex<Double>(Double.class, "i");
        
        a.addToAdjList(b, 4.0);
        a.addToAdjList(h, 8.0);
        
        b.addToAdjList(c, 8.0);
        b.addToAdjList(h, 11.0);
        
        c.addToAdjList(d, 7.0);
        c.addToAdjList(f, 4.0);
        c.addToAdjList(i, 2.0);
        
        d.addToAdjList(e, 9.0);
        d.addToAdjList(f, 14.0);
        
        e.addToAdjList(f, 10.0);

        f.addToAdjList(g, 2.0);
        
        g.addToAdjList(h, 1.0);
        g.addToAdjList(i, 6.0);
        
        h.addToAdjList(i, 7.0);
        
        G.addVertex(a);
        G.addVertex(b);
        G.addVertex(c);
        G.addVertex(d);
        G.addVertex(e);
        G.addVertex(f);
        G.addVertex(g);
        G.addVertex(h);
        G.addVertex(i);
        
        /*for (Vertex<Double> v : G.vertices) {
            System.out.printf("Hash code for %s: %d\n", v.name, v.hashCode());
        }*/
        
        LinkedList<Vertex<Double>> O = topologicalOrder(G);
        System.out.println("Topological order of the graph G:");
        int cnt = 0;
        for (Vertex<Double> v : O) {
            if (cnt++ > 0) {
                System.out.println('v');
            }
            System.out.println(v.name);
        }
    }
    
    /**
     * Compute the minimum spanning tree of the graph G by Prim's algorithm.
     * 
     * @param G
     * @return
     */
    public static <K> LinkedList<Edge<K>> minimumSpanningTreePrim(Graph<K> G) {
        Vertex<K> r = G.vertices.element();
        final Map<Vertex<K>, Double> S = new HashMap<Vertex<K>, Double>();
        for (Vertex<K> v : G.vertices) {
            S.put(v, Double.POSITIVE_INFINITY);
        }
        S.put(r, 0.0);
        PriorityQueue<Vertex<K>> Q = new PriorityQueue<Vertex<K>>(
                G.vertices.size(), 
                new Comparator<Vertex<K>>() {
                    @Override
                    public int compare(Vertex<K> v1, Vertex<K> v2) {
                        return S.get(v2).compareTo(S.get(v1));
                    }
                }
                );
        for (Vertex<K> v : G.vertices) {
            Q.insert(v);
        }
        while (!Q.isEmpty()) {
            Vertex<K> u = Q.poll();
            /*System.out.println(u);
            System.out.println(Q.keyToIndexMap.get(u));
            System.out.println(u.key);
            System.out.println(u.parent);
            System.out.println(u.adjcencyList);*/
            /*for (Pair<Vertex<K>, K> edge : u.adjcencyList) {
                Vertex<K> v = edge.first;
                K w = edge.second;*/
            for (Entry<Vertex<K>, Double> entry : u.adjacencyMap.entrySet()) {
                Vertex<K> v = entry.getKey();
                Double w = entry.getValue();
                /*System.out.println(v);
                System.out.println(Q.keyToIndexMap.get(v));
                System.out.println(Q.contains(v));
                System.out.println(w.compareTo(v.key));*/
                if (Q.contains(v) && w.compareTo(S.get(v)) < 0) {
                    v.parent = u;
                    S.put(v, w);
                    Q.heapify(v);
                }
            }
        }
        
        LinkedList<Edge<K>> MST = new LinkedList<Edge<K>>();
        for (Vertex<K> v : G.vertices) {
            if (v != r) {
                MST.add(new Edge<K>(v.parent, v, S.get(v)));
            }
        }
        return MST;
    }
    
    /**
     * Compute the shortest path from s to all other vertices in the graph G by 
     * Dijkstra's algorithm.
     * 
     * @param G
     * @param s
     * @return
     */
    public static <K> Map<Vertex<K>, Double> shortestPathDijkstra(Graph<K> G, Vertex<K> s) {
        // S[v] is v.d i.e., the upper bound for delta(s, v)
        final Map<Vertex<K>, Double> S = new HashMap<Vertex<K>, Double>();
        for (Vertex<K> v : G.vertices) {
            S.put(v, Double.POSITIVE_INFINITY);
        }
        S.put(s, 0.0);
        PriorityQueue<Vertex<K>> Q = new PriorityQueue<Vertex<K>>(
                G.vertices.size(), 
                new Comparator<Vertex<K>>() {
                    @Override
                    public int compare(Vertex<K> v1, Vertex<K> v2) {
                        return S.get(v2).compareTo(S.get(v1));
                    }
                }
                );
        for (Vertex<K> v : G.vertices) {
            Q.insert(v);
        }
        while (!Q.isEmpty()) {
            Vertex<K> u = Q.poll();
            for (Entry<Vertex<K>, Double> entry : u.adjacencyMap.entrySet()) {
                Vertex<K> v = entry.getKey();
                double w = entry.getValue();
                double w2 = S.get(u) + w;
                if (w2 < S.get(v)) {
                    v.parent = u;
                    S.put(v, w2);
                    Q.heapify(v);
                }   
            }
        }
        TreeMap<Vertex<K>, Double> res = new TreeMap<Vertex<K>, Double>(
                new Comparator<Vertex<K>>() {
                    @Override
                    public int compare(Vertex<K> o1, Vertex<K> o2) {
                        return S.get(o1).compareTo(S.get(o2));
                    }
                    
                });
        res.putAll(S);
        return res;
    }
    
    /**
     * Compute the topological order of a directed acyclic graph.
     * 
     * @param G
     * @return
     */
    public static <K> LinkedList<Vertex<K>> topologicalOrder(Graph<K> G) {
    	LinkedList<Vertex<K>> res = new LinkedList<Vertex<K>>();
		HashMap<Vertex<K>, Integer> countMap = new HashMap<Vertex<K>, Integer>();
		for (Vertex<K> u : G.vertices) {
			countMap.put(u, 0);
		}
		for (Vertex<K> u : G.vertices) {
			for (Vertex<K> v : u.adjacencyMap.keySet()) {
				countMap.put(v, countMap.get(v) + 1);
			}
		}
		LinkedList<Vertex<K>> queue = new LinkedList<Vertex<K>>();
		for (Map.Entry<Vertex<K>, Integer> entry : countMap.entrySet())
			if (entry.getValue() == 0)
				queue.add(entry.getKey());
		while (!queue.isEmpty()) {
			Vertex<K> u = queue.poll();
			res.add(u);
			for (Vertex<K> v : u.adjacencyMap.keySet()) {
				int count = countMap.get(v) - 1;
				countMap.put(v, count);
				if (count == 0) {
					queue.add(v);
				}
			}
		}
		if (res.size() < G.vertices.size()) {
			System.err.println("The graph has a cycle.");
			return null;
		}
		return res;
    }

}
