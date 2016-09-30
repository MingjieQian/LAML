package la.matrix;

import la.vector.Vector;

public interface Matrix {
	
	/**
	 * Get number of rows.
	 * 
	 * @return M
	 */
	public int getRowDimension();
	
	/**
	 * Get number of columns.
	 * 
	 * @return N
	 */
	public int getColumnDimension();
	
	/**
	 * Get the (r, c) entry.
	 * 
	 * @param r row index
	 * 
	 * @param c column index
	 * 
	 * @return this(r, c)
	 */
	public double getEntry(int r, int c);
	
	/**
	 * Note that it will be slow for sparse matrices, 
	 * even if bisection search is used.
	 * 
	 * @param r row index
	 * 
	 * @param c column index
	 * 
	 * @param v value to set
	 */
	public void setEntry(int r, int c, double v);
	
	/**
     * Get a submatrix. Row index and column index start from 0.
     *
     * @param startRow Initial row index
     * 
     * @param endRow Final row index (inclusive)
     * 
     * @param startColumn Initial column index
     * 
     * @param endColumn Final column index (inclusive)
     * 
     * @return The subMatrix containing the data of the
     *         specified rows and columns
     */
	public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn);
	
	/**
	 * Get a submatrix. Row index and column index start from 0.
	 *
	 * @param selectedRows Array of row indices
	 * 
	 * @param selectedColumns Array of column indices
	 * 
	 * @return The subMatrix containing the data in the
	 *         specified rows and columns
	 */
	public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns);
		
	/**
	 * Get rows from startRow to endRow as a matrix.
	 * 
	 * @param startRow start index
	 * 
	 * @param endRow end index (inclusive)
	 * 
	 * @return a real matrix containing the specified rows
	 */
	public Matrix getRows(int startRow, int endRow);
	
	/**
	 * Get rows of specified row indices as a matrix.
	 *  
	 * @param selectedRows a sequence of row indices
	 * 
	 * @return a real matrix containing the specified rows
	 */
	public Matrix getRows(int ... selectedRows);
	
	/**
	 * Get rows from startRow to endRow as a 1D {@code Vector} array.
	 * 
	 * @param startRow start index
	 * 
	 * @param endRow end index (inclusive)
	 * 
	 * @return a 1D {@code Vector} array containing the specified rows
	 */
	public Vector[] getRowVectors(int startRow, int endRow);
	
	/**
	 * Get rows of specified row indices as a 1D {@code Vector} array.
	 * 
	 * @param selectedRows a sequence of row indices
	 * 
	 * @return a 1D {@code Vector} array containing the specified rows
	 */
	public Vector[] getRowVectors(int ... selectedRows);
	
	/**
	 * Get a row matrix.
	 * 
	 * @param r row index
	 * 
	 * @return the r-th row matrix
	 */
	public Matrix getRowMatrix(int r);
	
	/**
	 * Set the r-th row by a row matrix A.
	 * 
	 * @param r row index
	 * 
	 * @param A a row matrix
	 */
	public void setRowMatrix(int r, Matrix A);
	
	/**
	 * Get a row vector.
	 * 
	 * @param r row index
	 * 
	 * @return the r-th row vector
	 */
	public Vector getRowVector(int r);
	
	/**
	 * Set the r-th row by a row vector V.
	 * 
	 * @param r row index
	 * 
	 * @param V a row vector
	 */
	public void setRowVector(int r, Vector V);
	
	/**
	 * Get columns from startColumn to endColumn as a matrix.
	 * 
	 * @param startColumn start index
	 * 
	 * @param endColumn end index (inclusive)
	 * 
	 * @return a real matrix containing the specified columns
	 */
	public Matrix getColumns(int startColumn, int endColumn);
	
	/**
	 * Get columns of specified column indices as a matrix.
	 *  
	 * @param selectedColumns a sequence of column indices
	 * 
	 * @return a real matrix containing the specified columns
	 */
	public Matrix getColumns(int ... selectedColumns);
	
	/**
	 * Get columns from startColumn to endColumn as a 1D {@code Vector} array.
	 * 
	 * @param startColumn start index
	 * 
	 * @param endColumn end index (inclusive)
	 * 
	 * @return a 1D {@code Vector} array containing the specified columns
	 */
	public Vector[] getColumnVectors(int startColumn, int endColumn);
	
	/**
	 * Get columns of specified column indices as a 1D {@code Vector} array.
	 * 
	 * @param selectedColumns a sequence of column indices
	 * 
	 * @return a 1D {@code Vector} array containing the specified columns
	 */
	public Vector[] getColumnVectors(int ... selectedColumns);
	
	/**
	 * Get a column matrix.
	 * 
	 * @param c column index
	 * 
	 * @return the c-th column matrix
	 */
	public Matrix getColumnMatrix(int c);
	
	/**
	 * Set the c-th column by a column matrix A.
	 * 
	 * @param c column index
	 * 
	 * @param A a column matrix
	 */
	public void setColumnMatrix(int c, Matrix A);
	
	/**
	 * Get a column vector.
	 * 
	 * @param c column index
	 * 
	 * @return the c-th column vector
	 */
	public Vector getColumnVector(int c);
	
	/**
	 * Set the c-th column by a column vector V.
	 * 
	 * @param c column index
	 * 
	 * @param V a column vector
	 */
	public void setColumnVector(int c, Vector V);
	
	/**
	 * This * A.
	 * 
	 * @param A a matrix
	 * 
	 * @return this * A
	 */
	public Matrix mtimes(Matrix A);
	
	/**
	 * This .* A.
	 * 
	 * @param A
	 * 
	 * @return this .* A
	 */
	public Matrix times(Matrix A);
	
	/**
	 * This .* v.
	 * 
	 * @param v a real scalar
	 * 
	 * @return this * v
	 */
	public Matrix times(double v);
	
	/**
	 * This + A.
	 * 
	 * @param A a real matrix
	 * 
	 * @return this + A
	 */
	public Matrix plus(Matrix A);
	
	/**
	 * This + v.
	 * 
	 * @param v v a real scalar
	 * 
	 * @return this + v
	 */
	public Matrix plus(double v);
	
	/**
	 * This - A.
	 * 
	 * @param A a real matrix
	 * 
	 * @return this - A
	 */
	public Matrix minus(Matrix A);
	
	/**
	 * This - v.
	 * 
	 * @param v a real scalar
	 * 
	 * @return this - v
	 */
	public Matrix minus(double v);
	
	/**
	 * The transpose of this matrix.
	 * 
	 * @return this<sup>T</sup>
	 */
	public Matrix transpose();
	
	/**
	 * Get a deep copy of this matrix.
	 * 
	 * @return a copy of this matrix
	 */
	public Matrix copy();

	/**
	 * Matrix vector operation, i.e., res = This * b.
	 * 
	 * @param b a dense or sparse vector
	 * 
	 * @return This * b
	 */
	public Vector operate(Vector b);
	
	/**
	 * Clear this matrix.
	 */
	public void clear();
	
	/**
	 * Get the 2D {@code double} array representation 
	 * of the matrix.
	 * 
	 * @return a 2D {@code double} array
	 */
	public double[][] getData();
	
}
