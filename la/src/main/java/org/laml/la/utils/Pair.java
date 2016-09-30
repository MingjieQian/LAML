package org.laml.la.utils;

public class Pair<A extends Comparable<? super A>,
B extends Comparable<? super B>>
implements Comparable<Pair<A, B>> {

	/*public final A first;
	public final B second;

	private Pair(A first, B second) {
		this.first = first;
		this.second = second;
	}*/
	
	public A first;
	public B second;
	
	public Pair(Pair<A, B> pair) {
		first = pair.first;
		second = pair.second;
	}

	public Pair(A first, B second) {
		this.first = first;
		this.second = second;
	}

	public static <A extends Comparable<? super A>,
	B extends Comparable<? super B>>
	Pair<A, B> of(A first, B second) {
		return new Pair<A, B>(first, second);
	}

	@Override
	public int compareTo(Pair<A, B> o) {
		int cmp = o == null ? 1 : (this.first).compareTo(o.first);
		return cmp == 0 ? (this.second).compareTo(o.second) : cmp;
	}

	@Override
	public int hashCode() {
		return 31 * hashcode(first) + hashcode(second);
	}

	private static int hashcode(Object o) {
		return o == null ? 0 : o.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Pair))
			return false;
		if (this == obj)
			return true;
		return equal(first, ((Pair<?, ?>) obj).first)
				&& equal(second, ((Pair<?, ?>) obj).second);
	}

	private boolean equal(Object o1, Object o2) {
		return o1 == o2 || (o1 != null && o1.equals(o2));
	}

	@Override
	public String toString() {
		return "(" + first + ", " + second + ')';
	}
	
	/**
	 * Return a shallow copy of this pair.
	 */
	@Override
	public Pair<A, B> clone() {
		return new Pair<A, B>(this);
	}
	
}
