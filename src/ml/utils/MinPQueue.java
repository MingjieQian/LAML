package ml.utils;

/*************************************************************************
 *  Compilation:  javac MinPQ.java
 *  Execution:    java MinPQ < input.txt
 *  
 *  Generic min priority queue implementation with a binary heap.
 *  Can be used with a comparator instead of the natural order.
 *
 *  % java MinPQ < tinyPQ.txt
 *  E A E (6 left on pq)
 *
 *  We use a one-based array to simplify parent and child calculations.
 *
 *  Can be optimized by replacing full exchanges with half exchanges
 *  (ala insertion sort).
 *
 *************************************************************************/

import java.util.HashMap;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 *  The <tt>MinPQ</tt> class represents a priority queue of generic keys.
 *  It supports the usual <em>insert</em> and <em>delete-the-minimum</em>
 *  operations, along with methods for peeking at the minimum key,
 *  testing if the priority queue is empty, and iterating through
 *  the keys.
 *  <p>
 *  This implementation uses a binary heap.
 *  The <em>insert</em> and <em>delete-the-minimum</em> operations take
 *  logarithmic amortized time.
 *  The <em>min</em>, <em>size</em>, and <em>is-empty</em> operations take constant time.
 *  Construction takes time proportional to the specified capacity or the number of
 *  items used to initialize the data structure.
 *  <p>
 *  For additional documentation, see <a href="http://algs4.cs.princeton.edu/24pq">Section 2.4</a> of
 *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 *  
 *  This modification is able to decrease the key of an element in the priority queue.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 *  @author Mingjie Qian
 * @param <K>
 */
public class MinPQueue<Key extends Comparable<Key> & Updater<K>, K> implements Iterable<Key> {
    private Key[] pq;                    // store items at indices 1 to N
    private int N;                       // number of items on priority queue
    
    private HashMap<Key, Integer> keyToIndexMap = new HashMap<Key, Integer>();

    /**
     * Initializes an empty priority queue with the given initial capacity.
     * @param initCapacity the initial capacity of the priority queue
     */
    @SuppressWarnings("unchecked")
	public MinPQueue(int initCapacity) {
        pq = (Key[]) new Comparable[initCapacity + 1];
        N = 0;
    }

    /**
     * Initializes an empty priority queue.
     */
    public MinPQueue() {
        this(1);
    }

    /**
     * Initializes a priority queue from the array of keys.
     * Takes time proportional to the number of keys, using sink-based heap construction.
     * @param keys the array of keys
     */
    @SuppressWarnings("unchecked")
	public MinPQueue(Key[] keys) {
        N = keys.length;
        pq = (Key[]) new Comparable[keys.length + 1];
        for (int i = 0; i < N; i++) {
            pq[i+1] = keys[i];
            keyToIndexMap.put(pq[i + 1], i + 1);
        }
        for (int k = N/2; k >= 1; k--)
            sink(k);
        assert isMinHeap();
    }

    /**
     * Is the priority queue empty?
     * @return true if the priority queue is empty; false otherwise
     */
    public boolean isEmpty() {
        return N == 0;
    }

    /**
     * Returns the number of keys on the priority queue.
     * @return the number of keys on the priority queue
     */
    public int size() {
        return N;
    }

    /**
     * Returns a smallest key on the priority queue.
     * @return a smallest key on the priority queue
     * @throws java.util.NoSuchElementException if priority queue is empty
     */
    public Key min() {
        if (isEmpty()) throw new NoSuchElementException("Priority queue underflow");
        return pq[1];
    }

    // helper function to double the size of the heap array
    @SuppressWarnings("unchecked")
	private void resize(int capacity) {
        assert capacity > N;
        Key[] temp = (Key[]) new Comparable[capacity];
        for (int i = 1; i <= N; i++) temp[i] = pq[i];
        pq = temp;
    }
    
    public boolean containsKey(Key key) {
    	return keyToIndexMap.containsKey(key);
    }

    /**
     * Adds a new key to the priority queue.
     * @param x the key to add to the priority queue
     */
    public void insert(Key x) {
        // double size of array if necessary
        if (N == pq.length - 1) resize(2 * pq.length);

        // add x, and percolate it up to maintain heap invariant
        pq[++N] = x;
        keyToIndexMap.put(x, N);
        swim(N);
        assert isMinHeap();
    }

    /**
     * Removes and returns a smallest key on the priority queue.
     * @return a smallest key on the priority queue
     * @throws java.util.NoSuchElementException if the priority queue is empty
     */
    public Key delMin() {
        if (isEmpty()) throw new NoSuchElementException("Priority queue underflow");
        exch(1, N);
        Key min = pq[N--];
        keyToIndexMap.remove(min);
        sink(1);
        pq[N+1] = null;         // avoid loitering and help with garbage collection
        if ((N > 0) && (N == (pq.length - 1) / 4)) resize(pq.length  / 2);
        assert isMinHeap();
        return min;
    }
    
    public void delete(Key x) {
    	if (!keyToIndexMap.containsKey(x))
    		return;
    	if (isEmpty()) throw new NoSuchElementException("Priority queue underflow");
    	int idx = keyToIndexMap.get(x);
    	exch(idx, N);
        N--;
        keyToIndexMap.remove(x);
        sink(idx);
        pq[N+1] = null;         // avoid loitering and help with garbage collection
        if ((N > 0) && (N == (pq.length - 1) / 4)) resize(pq.length  / 2);
        assert isMinHeap();
    }

    public void update(Key key, K k) {
    	int index = keyToIndexMap.get(key);
    	key.update(k);
    	if (index > 1 && greater(index / 2, index))
    		swim(index);
    	else
    		sink(index);
    	
    }

   /***********************************************************************
    * Helper functions to restore the heap invariant.
    **********************************************************************/

    private void swim(int k) {
        while (k > 1 && greater(k/2, k)) {
            exch(k, k/2);
            k = k/2;
        }
    }

    private void sink(int k) {
        while (2*k <= N) {
            int j = 2*k;
            if (j < N && greater(j, j+1)) j++;
            if (!greater(k, j)) break;
            exch(k, j);
            k = j;
        }
    }

   /***********************************************************************
    * Helper functions for compares and swaps.
    **********************************************************************/
    private boolean greater(int i, int j) {
    	return pq[i].compareTo(pq[j]) > 0;
    }

    private void exch(int i, int j) {
    	Key swap = pq[i];
        pq[i] = pq[j];
        pq[j] = swap;
        // Modified on Dec. 24th, 2014 
        // {
        keyToIndexMap.put(pq[i], i);
    	keyToIndexMap.put(pq[j], j);
    	// }
    }

    // is pq[1..N] a min heap?
    private boolean isMinHeap() {
        return isMinHeap(1);
    }

    // is subtree of pq[1..N] rooted at k a min heap?
    private boolean isMinHeap(int k) {
        if (k > N) return true;
        int left = 2*k, right = 2*k + 1;
        if (left  <= N && greater(k, left))  return false;
        if (right <= N && greater(k, right)) return false;
        return isMinHeap(left) && isMinHeap(right);
    }


   /***********************************************************************
    * Iterators
    **********************************************************************/

    /**
     * Returns an iterator that iterates over the keys on the priority queue
     * in ascending order.
     * The iterator doesn't implement <tt>remove()</tt> since it's optional.
     * @return an iterator that iterates over the keys in ascending order
     */
    public Iterator<Key> iterator() { return new HeapIterator(); }

    private class HeapIterator implements Iterator<Key> {
        // create a new pq
        private MinPQueue<Key, K> copy;

        // add all items to copy of heap
        // takes linear time since already in heap order so no keys move
        public HeapIterator() {
            copy = new MinPQueue<Key, K>(size());
            for (int i = 1; i <= N; i++)
                copy.insert(pq[i]);
        }

        public boolean hasNext()  { return !copy.isEmpty();                     }
        public void remove()      { throw new UnsupportedOperationException();  }

        public Key next() {
            if (!hasNext()) throw new NoSuchElementException();
            return copy.delMin();
        }
    }

}
