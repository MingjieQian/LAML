package la.vector;

import la.matrix.Matrix;

public interface Vector {
	
	/**
	 * Get the dimensionality of this vector.
	 * 
	 * @return dimensionality of this vector
	 */
	public int getDim();
	
	/**
	 * Get a deep copy of this vector.
	 * 
	 * @return a copy of this vector
	 */
	public Vector copy();
	
	/**
	 * Element-wise multiplication, i.e. res = this .* V.
	 * 
	 * @param V a real vector
	 * 
	 * @return this .* V
	 */
	public Vector times(Vector V);
	
	/**
	 * res = v * this.
	 * 
	 * @param v a real scalar
	 * 
	 * @return v * this
	 */
	public Vector times(double v);
	
	/**
	 * Vector addition, i.e. res = this + V.
	 * 
	 * @param V a real vector
	 * 
	 * @return this + V
	 */
	public Vector plus(Vector V);
	
	/**
	 * Vector subtraction, i.e. res = this - V.
	 * 
	 * @param V a real vector
	 * 
	 * @return this - V
	 */
	public Vector minus(Vector V);
	
	/**
	 * Get the value of the i-th entry.
	 * 
	 * @param i index
	 * 
	 * @return this(i)
	 */
	public double get(int i);
	
	/**
	 * Set the value of the i-th entry.
	 * 
	 * @param i index
	 * 
	 * @param v value to set
	 */
	public void set(int i, double v);
	
	/**
	 * Vector matrix multiplication, i.e. res = this' * A.
	 * 
	 * @param A a real matrix
	 * 
	 * @return this<sup>T</sup> * A
	 */
	public Vector operate(Matrix A);
	
	/**
	 * Clear this vector so that all entries are zero.
	 */
	public void clear();
	
}
