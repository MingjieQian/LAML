package la.utils;

public abstract class Node<T extends Comparable<T>> {
	
	public T key;
	
	public T getKey() {
		return this.key;
	}
	
	public abstract Node<T> getLeft();
	public abstract Node<T> getRight();
	
	// public Node<T> p;
	/*public Node<T> left;
	public Node<T> right;*/
	// public Color color;
	
	/*public Node(T key, Color color) {
		this.key = key;
		this.color = color;
		p = null;
		left = null;
		right = null;
	}*/
	
	public Node(T key) {
        this.key = key;
        /*this.left = null;
        this.right = null;*/
    }
	
	@Override
    public String toString() {
        if (key == null) {
            return "null";
        } else {
            return key.toString();
        }
    }
	
	/*@Override
	public String toString() {
		
		String res = "";
		if (this != null) {
			res = "Key: " + key.toString();
			if (p != null && p.left != null)
				res += System.getProperty("line.separator") + "p.key: " + this.p.key.toString();
			if (left != null && left.left != null)
				res += System.getProperty("line.separator") + "left.key: " + this.left.key.toString();
			if (right != null && right.right != null)
				res += System.getProperty("line.separator") + "right.key: " + this.right.key.toString();
		}
			
		return res;
		
	}*/
	
}
