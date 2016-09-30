package org.laml.ml.graph;

public class Edge<K> {
    
    Vertex<K> u;
    
    Vertex<K> v;
    
    double weight;

    public Edge(Vertex<K> u, Vertex<K> v, double weight) {
        this.u = u;
        this.v = v;
        this.weight = weight;
    }
    
    @Override
    public String toString() {
        return String.format("%s -- %s: %s", u.name, v.name, weight);
    }
    
}
