package ml.utils;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * The <tt>PriorityQueue</tt> class represents a priority queue of generic elements.
 * It supports the usual <em>insert</em> and <em>poll</em>
 * operations, along with methods for peeking at the top element,
 * testing if the priority queue is empty, and iterating through
 * the elements.
 * <p>
 * This implementation uses a binary heap.
 * The <em>insert</em> and <em>poll</em> operations take
 * logarithmic amortized time.
 * The <em>peek</em>, <em>size</em>, and <em>is-empty</em> operations take constant time.
 * Construction takes time proportional to the specified capacity or the number of
 * items used to initialize the data structure.
 * <p>
 * For additional documentation, see <a href="http://algs4.cs.princeton.edu/24pq">Section 2.4</a> of
 * <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 *
 * This modification is able to adjust the element in the priority queue to heapify the 
 * inner array.
 *  
 * @author Mingjie Qian
 * @version 1.0 March 24th, 2016
 * @param <E>
 */
public class PriorityQueue<E> implements Iterable<E> {
    private E[] pq;                      // store items at indices 1 to N
    private int N;                       // number of items on priority queue
    
    private final Comparator<E> comparator;
    
    private HashMap<E, Integer> keyToIndexMap = new HashMap<E, Integer>();

    /**
     * Initializes an empty priority queue with the given initial capacity and
     * comparator.
     * 
     * If the comparator is implemented such hat the first argument is less than 
     * the second argument, then this is a maximum priority queue.
     *
     * @param initCapacity
     * @param comparator
     */
    @SuppressWarnings("unchecked")
    public PriorityQueue(int initCapacity, Comparator<E> comparator) {
        pq = (E[]) new Object[initCapacity + 1];
        N = 0;
        this.comparator = comparator;
    }
    
    /**
     * Initializes an empty priority queue with the given initial capacity.
     * @param initCapacity the initial capacity of the priority queue
     */
    public PriorityQueue(int initCapacity) {
        this(initCapacity, null);
    }

    /**
     * Initializes an empty priority queue.
     */
    public PriorityQueue() {
        this(1);
    }

    /**
     * Initializes a priority queue from the array of elements.
     * Takes time proportional to the number of elements, using sink-based heap construction.
     * @param elements the array of elements
     */
    @SuppressWarnings("unchecked")
    public PriorityQueue(E[] elements) {
        N = elements.length;
        pq = (E[]) new Object[elements.length + 1]; 
        for (int i = 0; i < N; i++) {
            pq[i+1] = elements[i];
            keyToIndexMap.put(pq[i + 1], i + 1);
        }
        for (int k = N/2; k >= 1; k--)
            sink(k);
        assert isMaxHeap();
        comparator = null;
    }

    /**
     * Is the priority queue empty?
     * @return true if the priority queue is empty; false otherwise
     */
    public boolean isEmpty() {
        return N == 0;
    }

    /**
     * Returns the number of elements on the priority queue.
     * @return the number of elements on the priority queue
     */
    public int size() {
        return N;
    }

    // helper function to double the size of the heap array
    @SuppressWarnings("unchecked")
    private void resize(int capacity) {
        assert capacity > N;
        E[] temp = (E[]) new Object[capacity];
        for (int i = 1; i <= N; i++) temp[i] = pq[i];
        pq = temp;
    }
    
    public boolean contains(E element) {
        return keyToIndexMap.containsKey(element);
    }

    /**
     * Adds a new key to the priority queue.
     * @param x the new key to add to the priority queue
     */
    public void insert(E x) {

        // double size of array if necessary
        if (N >= pq.length - 1) resize(2 * pq.length);

        // add x, and percolate it up to maintain heap invariant
        pq[++N] = x;
        keyToIndexMap.put(x, N);
        swim(N);
        // assert isMaxHeap();
    }
    
    /**
     * Returns the top element on the priority queue.
     * @return the top element on the priority queue
     * @throws java.util.NoSuchElementException if the priority queue is empty
     */
    public E peek() {
        if (isEmpty()) throw new NoSuchElementException("Priority queue underflow");
        return pq[1];
    }

    /**
     * Removes and returns the top element on the priority queue.
     * @return the top element on the priority queue
     * @throws java.util.NoSuchElementException if priority queue is empty.
     */
    public E poll() {
        if (isEmpty()) throw new NoSuchElementException("Priority queue underflow");
        E max = pq[1];
        exch(1, N--);
        keyToIndexMap.remove(max);
        sink(1);
        pq[N+1] = null;     // to avoid loiterig and help with garbage collection
        if ((N > 0) && (N == (pq.length - 1) / 4)) resize(pq.length / 2);
        // assert isMaxHeap();
        return max;
    }
    
    public void delete(E x) {
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
        // assert isMaxHeap();
    }

    /**
     * Establishes the heap invariant for an element.
     * @param element
     */
    public void heapify(E element) {
        int index = keyToIndexMap.get(element);
        if (index > 1 && less(index / 2, index))
            swim(index);
        else
            sink(index);
    }

   /***********************************************************************
    * Helper functions to restore the heap invariant.
    **********************************************************************/

    private void swim(int k) {
        while (k > 1 && less(k/2, k)) {
            exch(k, k/2);
            k = k/2;
        }
    }

    private void sink(int k) {
        while (2*k <= N) {
            int j = 2*k;
            if (j < N && less(j, j+1)) j++;
            if (!less(k, j)) break;
            exch(k, j);
            k = j;
        }
    }

   /***********************************************************************
    * Helper functions for compares and swaps.
    **********************************************************************/
    @SuppressWarnings("unchecked")
    private boolean less(int i, int j) {
        if (comparator != null) {
            return comparator.compare(pq[i], pq[j]) < 0;
        } else {
            return ((Comparable<? super E>) pq[i]).compareTo(pq[j]) < 0;
        }
    }

    private void exch(int i, int j) {
        E swap = pq[i];
        pq[i] = pq[j];
        pq[j] = swap;
        // Modified on Dec. 24th, 2014 
        // {
        keyToIndexMap.put(pq[i], i);
        keyToIndexMap.put(pq[j], j);
        // }
    }

    // is pq[1..N] a max heap?
    private boolean isMaxHeap() {
        return isMaxHeap(1);
    }

    // is subtree of pq[1..N] rooted at k a max heap?
    private boolean isMaxHeap(int k) {
        if (k > N) return true;
        int left = 2*k, right = 2*k + 1;
        if (left  <= N && less(k, left))  return false;
        if (right <= N && less(k, right)) return false;
        return isMaxHeap(left) && isMaxHeap(right);
    }


   /***********************************************************************
    * Iterator
    **********************************************************************/

    /**
     * Returns an iterator that iterates over the elements on the priority queue
     * in descending order.
     * The iterator doesn't implement <tt>remove()</tt> since it's optional.
     * @return an iterator that iterates over the elements in descending order
     */
    public Iterator<E> iterator() { return new HeapIterator(); }

    private class HeapIterator implements Iterator<E> {

        // create a new priority queue
        private PriorityQueue<E> copy;

        // add all items to copy of heap
        // takes linear time since already in heap order so no elements move
        public HeapIterator() {
            copy = new PriorityQueue<E>(size());
            for (int i = 1; i <= N; i++)
                copy.insert(pq[i]);
        }

        public boolean hasNext()  { return !copy.isEmpty();                     }
        public void remove()      { throw new UnsupportedOperationException();  }

        public E next() {
            if (!hasNext()) throw new NoSuchElementException();
            return copy.poll();
        }
    }

}
