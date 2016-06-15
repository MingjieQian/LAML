package ml.utils;

public class Entry <K extends Comparable<K>, V> implements Comparable<Entry<K, V>> {

	public K key;
	
	public V value;
	
	public Entry(K key, V value) {
		this.key = key;
		this.value = value;
	}
	
	@Override
	public String toString() {
		if (value == null) {
			return key.toString();
		} else {
			return "<" + key.toString() + "," + value.toString() + ">";
		}
	}

	@Override
	public int compareTo(Entry<K, V> o) {
		return this.key.compareTo(o.key);
	}

}
