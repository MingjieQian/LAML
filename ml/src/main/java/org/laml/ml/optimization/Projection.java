package org.laml.ml.optimization;

import org.laml.la.matrix.Matrix;

/**
 * Compute proj_tC(X), which is defined by
 * proj_tC(x) := argmin_{u \in tC} 1/2 * || u - x ||^2. 
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 25th, 2014
 */
public interface Projection {
	
	/**
	 * Compute proj_tC(X).
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return proj_tC(X)
	 */
	public Matrix compute(double t, Matrix X);
	
	/**
	 * res = proj_tC(X).
	 * 
	 * @param res result matrix
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 */
	public void compute(Matrix res, double t, Matrix X);

}
