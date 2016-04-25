package ml.graph;

import java.util.HashMap;

public class Vertex<K> {
    
    String name;
    
    K key;
    
    Vertex<K> parent;
    
    // LinkedList<Pair<Vertex<K>, K>> adjcencyList;
    
    /**
     * w(this, v) = adjcencyMap.get(v)
     */
    public HashMap<Vertex<K>, Double> adjacencyMap;
    
    Class<K> typeClass;
    
    @SuppressWarnings("unchecked")
    public Vertex(K key, String name) {
        this.name = name;
        this.key = key;
        typeClass = (Class<K>) key.getClass();
        parent = null;
        // adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
        adjacencyMap = new HashMap<Vertex<K>, Double>();
    }
    
    public Vertex(K key) {
        this(key, "");
    }
    
    public Vertex(Vertex<K> v) {
        this(v.key, v.name);
    }
    
    @SuppressWarnings("unchecked")
    public Vertex(Class<K> typeClass, String name) {
        this.typeClass = typeClass;
        if (typeClass == null) 
            this.key = null;
        else if (typeClass.equals(Double.class))
            this.key = (K) new Double(Double.POSITIVE_INFINITY);
        else if (typeClass.equals(Integer.class))
            this.key = (K) new Integer(Integer.MAX_VALUE);
        else if (typeClass.equals(Float.class))
            this.key = (K) new Float(Float.POSITIVE_INFINITY);
        else
            this.key = null;
        this.name = name;
        parent = null;
        // adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
        adjacencyMap = new HashMap<Vertex<K>, Double>();
    }
    
    public Vertex(Class<K> typeClass) {
        this(typeClass, "");
    }
    
    public Vertex() {
        this((Class<K>)null);
    }
    
    public Vertex(String name) {
        this((Class<K>)null, name);
    }
    
    public void addToAdjList(Vertex<K> v, double w) {
        // adjcencyList.add(Pair.of(v, w));
        adjacencyMap.put(v, w);
    }
    
    @Override
    public String toString() {
        if (name.isEmpty())
            return super.toString();
        else {
        	return key == null ? name : name + ':' + key;
        }
    }

}
