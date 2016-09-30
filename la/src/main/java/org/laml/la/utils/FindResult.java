package org.laml.la.utils;

/**
 * <h4>A wrapper for the output of find function.</h4>
 * There are three data members:<br/>
 * rows: row indices array for non-zero elements of a matrix<br/>
 * cols: column indices array for non-zero elements of a matrix<br/>
 * vals: values array for non-zero elements of a matrix<br/>
 */
public class FindResult {
	
	public int[] rows;
	public int[] cols;
	public double[] vals;
	
	public FindResult(int[] rows, int[] cols, double[] vals) {
		this.rows = rows;
		this.cols = cols;
		this.vals = vals;
	}

}
