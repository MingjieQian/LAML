package ml.graph;

import java.util.HashMap;
import java.util.LinkedList;

import ml.utils.Pair;
import ml.utils.Updater;

public class Vertex<K extends Comparable<K>> implements Comparable<Vertex<K>>, Updater<K> {
	
	String name;
	
	K key;
	
	Vertex<K> parent;
	
	LinkedList<Pair<Vertex<K>, K>> adjcencyList;
	
	/**
	 * w(this, v) = adjcencyMap.get(v)
	 */
	HashMap<Vertex<K>, K> adjcencyMap;
	
	Class<K> typeClass;
	
	@SuppressWarnings("unchecked")
	public Vertex(K key, String name) {
		this.name = name;
		this.key = key;
		typeClass = (Class<K>) key.getClass();
		parent = null;
		adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
		adjcencyMap = new HashMap<Vertex<K>, K>();
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
		adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
		adjcencyMap = new HashMap<Vertex<K>, K>();
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
	
	@SuppressWarnings("unchecked")
	public void addToAdjList(Vertex<K> v, K w) {
		if (typeClass == null)
			typeClass = (Class<K>) w.getClass();
		adjcencyList.add(Pair.of(v, w));
		adjcencyMap.put(v, w);
	}
	
	@Override
	public String toString() {
		if (name.isEmpty())
			return super.toString();
		else {
			return name + ':' + key;
		}
	}
	
	/*@Override
	public int hashCode() {
		return key.hashCode();
	}*/

	/*@Override
	public void decrease(K key) {
		if (this.key.compareTo(key) < 0) {
			System.err.println("Key should be decreased.");
			System.exit(1);
		}
		this.key = key;
	}*/

	@Override
	public int compareTo(Vertex<K> o) {
		int cmp = 0;
		if (o == null) {
			cmp = 1;
		} else if (key == null) {
			cmp = o.key == null ? 0 : -1;
		} else if (o.key == null) {
			cmp = key == null ? 0 : 1;
		} else {
			cmp = this.key.compareTo(o.key);
		}
		return cmp;
		// int cmp = o == null ? 1 : (this.key).compareTo(o.key);
		// return cmp == 0 ? (this.key).compareTo(o.key) : cmp;
	}

	@Override
	public void update(K key) {
		this.key = key;
	}

}
