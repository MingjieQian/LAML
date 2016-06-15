package ml.utils;

public class BinaryNode<T extends Comparable<T>> extends Node<T> {

	public BinaryNode<T> left;
	public BinaryNode<T> right;
	
	public BinaryNode(T key) {
		super(key);
		left = null;
		right = null;
	}

	@Override
	public BinaryNode<T> getLeft() {
		return left;
	}

	@Override
	public BinaryNode<T> getRight() {
		return right;
	}

}
