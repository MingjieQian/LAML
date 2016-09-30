package org.laml.ml.subspace;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;

public abstract class DimensionalityReduction {
	
	/**
	 * n x d data matrix.
	 */
	protected Matrix X;
	
	/**
	 * Reduced n x r data matrix.
	 */
	protected Matrix R;
	
	/**
	 * Reduced dimensionality.
	 */
	protected int r;
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public DimensionalityReduction(int r) {
		this.r = r;
	}
	
	/**
	 * Do dimensionality reduction.
	 */
	public abstract void run();
	
	/**
	 * 
	 * @param X an nExample x nFeature data matrix with each row being
	 *             a data example
	 */
	public void feedData(Matrix X) {
		this.X = X;
	}
	
	/**
	 * Feed training data for this dimensionality reduction algorithm.
	 * 
	 * @param data an nExample x nFeature 2D {@code double} array with each
	 *             row being a data sample
	 */
	public void feedData(double[][] data) {
		this.X = new DenseMatrix(data);
	}
	
	public Matrix getDataMatrix() {
		return X;
	}
	
	public Matrix getReducedDataMatrix() {
		return R;
	}
	
	public void setReducedDimensionality(int r) {
		this.r = r;
	}
	
	public int getReducedDimensionality() {
		return r;
	}

}
