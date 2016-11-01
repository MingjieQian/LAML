package org.laml.ml.graph;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import static org.laml.la.utils.Printer.*;

/**
 * Maximum flow.
 * 
 * @author Mingjie Qian
 * @version 1.0 April 24th, 2016
 */
public class MaximumFlow {

	public static void main(String[] args) {
		Vertex<Object> s = new Vertex<Object>("s");
		Vertex<Object> v1 = new Vertex<Object>("v1");
		Vertex<Object> v2 = new Vertex<Object>("v2");
		Vertex<Object> v3 = new Vertex<Object>("v3");
		Vertex<Object> v4 = new Vertex<Object>("v4");
		Vertex<Object> t = new Vertex<Object>("t");
		
		s.addToAdjList(v1, 16);
		s.addToAdjList(v2, 13);
		
		v1.addToAdjList(v3, 12);
		
		v2.addToAdjList(v1, 4);
		v2.addToAdjList(v4, 14);
		
		v3.addToAdjList(v2, 9);
		v3.addToAdjList(t, 20);
		
		v4.addToAdjList(v3, 7);
		v4.addToAdjList(t, 4);
		
		Graph<Object> G = new Graph<Object>();
		G.addVertex(s);
		G.addVertex(v1);
		G.addVertex(v2);
		G.addVertex(v3);
		G.addVertex(v4);
		G.addVertex(t);
		
		println("Flow of G:");
		HashMap<Vertex<Object>, HashMap<Vertex<Object>, Double>> f = FordFulkerson(G, s, t);
		for (Vertex<Object> u : f.keySet()) {
			if (f.get(u).isEmpty())
				continue;
			for (Entry<Vertex<Object>, Double> entry : f.get(u).entrySet()) {
				Vertex<Object> v = entry.getKey();
				double fuv = entry.getValue();
				printf("%s -> %s: %s\n", u, v, fuv);
			}
			println();
		}
	}
	
	/**
	 * The Ford-Fulkerson algorithm to solve the maximum flow problem.
	 * 
	 * @param G
	 * @param s
	 * @param t
	 * @return flow of G, f(u, v) is defined only if u -> v \in G.E
	 */
	public static <K> HashMap<Vertex<K>, HashMap<Vertex<K>, Double>> 
	FordFulkerson(Graph<K> G, Vertex<K> s, Vertex<K> t) {
		/* 
		 * Residual network of G induced by f is Gf = (V, Ef) where Ef is defined by
		 * Ef := {(u, v) \in V x V : cf(u, v) > 0}.
		 * 
		 * Gf[u] is the adjacency map of u, i.e., w(u, v) = Gf[u].get(v)
		 * 
		 * The residual capacity cf(u, v) is defined by
		 *           / c(u, v) - f(u, v) if u -> v \in G.E
		 * cf(u, v) =  f(v, u)			 if v -> u \in G.E
		 *           \ 0				 otherwise
		 */
		HashMap<Vertex<K>, HashMap<Vertex<K>, Double>> Gf = null;
		Gf = new HashMap<Vertex<K>, HashMap<Vertex<K>, Double>>();
		for (Vertex<K> u : G.vertices) {
			HashMap<Vertex<K>, Double> adjacencyMap = new HashMap<Vertex<K>, Double>();
			for (Entry<Vertex<K>, Double> entry : u.adjacencyMap.entrySet()) {
				Vertex<K> v = entry.getKey();
				double w = entry.getValue();
				if (w > 0)
					adjacencyMap.put(v, w);
			}
			Gf.put(u, adjacencyMap);
		}
		
		// Flow of G. f(u, v) is defined only if u -> v \in G.E
		HashMap<Vertex<K>, HashMap<Vertex<K>, Double>> f = null;
		f = new HashMap<Vertex<K>, HashMap<Vertex<K>, Double>>();
		for (Vertex<K> u : G.vertices) {
			HashMap<Vertex<K>, Double> adjacencyMap = new HashMap<Vertex<K>, Double>();
			for (Entry<Vertex<K>, Double> entry : u.adjacencyMap.entrySet()) {
				Vertex<K> v = entry.getKey();
				double w = 0;
				adjacencyMap.put(v, w);
			}
			f.put(u, adjacencyMap);
		}
		
		while (true) {
			List<Vertex<K>> p = findPath(Gf, s, t);
			if (p == null)
				break;
			// Compute cf(p) = min {cf(u, v) : u -> v \in p}
			double cfp = Double.MAX_VALUE;
			Vertex<K> u = null;
			for (Vertex<K> v : p) {
				if (u == null) {
					u = v;
					continue;
				}
				// Now we have an edge u -> v in the path
				double cf = Gf.get(u).get(v);
				if (cfp > cf) {
					cfp = cf;
				}
				u = v;
			}
			// Update Gf and f by cf(p)
			u = null;
			for (Vertex<K> v : p) {
				if (u == null) {
					u = v;
					continue;
				}
				// Now we have an edge u -> v in the path
				if (u.adjacencyMap != null && u.adjacencyMap.containsKey(v)) {
					// u -> v \in G.E, v -> u is the anti-edge of u -> v in G.E
					
					// Update cf for u -> v in Gf
					// Since cf(u, v) >= cfp > 0, we must have u -> v in Gf
					double cf = -1;
					cf = Gf.get(u).get(v) - cfp;
					if (cf > 0)
						Gf.get(u).put(v, cf);
					else
						Gf.get(u).remove(v);
					
					// Update cf for v -> u in Gf
					/*
					 * Since f(u, v) might be zero, cf(v, u) might be zero,
					 * thus edge v -> u may not exist in Gf.
					 */
					if (Gf.get(v).containsKey(u))
						cf = Gf.get(v).get(u) + cfp;
					else
						cf = cfp;
					// Actually cf must be larger than zero
					Gf.get(v).put(u, cf);
					
					// Update f(u -> v)
					f.get(u).put(v, f.get(u).get(v) + cfp);
				} else {
					// v -> u \in G.E, u -> v is the anti-edge of v -> u in G.E
					
					// Update cf for v -> u in Gf
					/*
					 * Since f(v, u) might be w(v, u), cf(v, u) might be zero,
					 * thus edge v -> u may not exist in Gf.
					 */
					double cf = -1;
					if (Gf.get(v).containsKey(u))
						cf = Gf.get(v).get(u) + cfp;
					else
						cf = cfp;
					// Actually cf must be greater than zero
					Gf.get(v).put(u, cf);
					
					// Update cf for u -> v in Gf
					// Since cf(u, v) >= cfp > 0, we must have u -> v in Gf
					cf = Gf.get(u).get(v) - cfp;
					if (cf > 0)
						Gf.get(u).put(v, cf);
					else
						Gf.get(u).remove(v);
				
					// Update f(v -> u)
					f.get(v).put(u, f.get(v).get(u) - cfp);
				}
				u = v;
			}
		}
		return f;
	}
	
	/**
	 * Find a simple path from source s to sink t from the residual network Gf.
	 * 
	 * @param Gf
	 * @param s
	 * @param t
	 * @return the simple path from s to t if there exists one, otherwise null
	 */
	public static <K> List<Vertex<K>> findPath(HashMap<Vertex<K>, HashMap<Vertex<K>, Double>> Gf, Vertex<K> s, Vertex<K> t) {
		List<Vertex<K>> path = null;
		Set<Vertex<K>> candidates = new HashSet<Vertex<K>>(Gf.keySet());
		Map<Vertex<K>, Vertex<K>> parentMap = new HashMap<Vertex<K>, Vertex<K>>();
		LinkedList<Vertex<K>> queue = new LinkedList<Vertex<K>>();
		queue.add(s);
		int thisLevelSize = 1;
		while (!queue.isEmpty()) {
			Vertex<K> u = queue.poll();
			thisLevelSize--;
			for (Vertex<K> v : Gf.get(u).keySet()) {
				if (!candidates.contains(v))
					continue;
				if (v == t) {
					// Build path from s to t: s ~> v
					path = new LinkedList<Vertex<K>>();
					path.add(0, t);
					Vertex<K> c = u;
					do {
						path.add(0, c);
						c = parentMap.get(c);
					} while (c != s);
					path.add(0, s);
					return path;
				}
				queue.add(v);
				parentMap.put(v, u);
				candidates.remove(v);
			}
			if (thisLevelSize == 0) {
				thisLevelSize = queue.size();
			}
		}
		return path;
	}

}
