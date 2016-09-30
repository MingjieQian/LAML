package la.matrix;

import static la.utils.Printer.err;
import static la.utils.Printer.sprintf;
import static la.utils.Utility.exit;

import java.io.Serializable;

import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import la.utils.ArrayOperator;

/***
 * A Java implementation of dense matrices.
 * 
 * @author Mingjie Qian
 * @version 1.0 Nov. 29th, 2013
 */
public class DenseMatrix implements Matrix, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6821454132254344419L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int M = 1000;
		int N = 1000;
		Matrix A = new DenseMatrix(M, N, 1);
		Matrix B = new DenseMatrix(M, N, 1);
		
		// 1000 x 1000 matrix multiplication test.
		// It takes 3.4 seconds in average.
		System.out.println("1000 x 1000 matrix multiplication test.");
		long start = System.currentTimeMillis();
		A.mtimes(B);
		System.out.format(System.getProperty("line.separator") + "Elapsed time: %.3f seconds.\n", 
				(System.currentTimeMillis() - start) / 1000F);	
		
	}

	/**
	 * Row dimension, i.e. number of rows.
	 */
	private int M;
	
	/**
	 * Column dimension, i.e. number of columns.
	 */
	private int N;
	
	private double[][] data;
	
	public DenseMatrix() {
		M = 0;
		N = 0;
		data = null;
	}
	
	/**
	 * Construct a 1 x 1 dense matrix with the given real number v.
	 * 
	 * @param v a real number
	 */
	public DenseMatrix(double v) {
		M = 1;
		N = 1;
		data = new double[][] {{v}};
	}

	/**
	 * Construct an M x N all-zero dense matrix.
	 * 
	 * @param M number of rows
	 * 
	 * @param N number of columns
	 */
	public DenseMatrix(int M, int N) {
		data = new double[M][];
		for (int i = 0; i < M; i++) {
			data[i] = new double[N];
			for (int j = 0; j < N; j++) {
				data[i][j] = 0.0;
			}
		}
		this.M = M;
		this.N = N;
	}
	
	/**
	 * Construct an all-zero dense matrix with a specified size.
	 * 
	 * @param size size[0] is the number of rows and size[1] is 
	 *             the number of columns
	 */
	public DenseMatrix(int[] size) {
		if (size.length != 2) {
			System.err.println("The input integer array should have exactly two entries!");
			System.exit(1);
		}
		int M = size[0];
		int N = size[1];
		data = new double[M][];
		for (int i = 0; i < M; i++) {
			data[i] = new double[N];
			for (int j = 0; j < N; j++) {
				data[i][j] = 0.0;
			}
		}
		this.M = M;
		this.N = N;
	}
	
	/**
	 * Construct a dense matrix given the data.
	 * 
	 * @param data a 2D {@code double} array
	 */
	public DenseMatrix(double[][] data) {
		this.data = data;
		this.M = data.length;
		this.N = M > 0 ? data[0].length : 0;
	}
	
	/**
	 * Construct a column or row matrix specified by
	 * a dimension parameter.
	 * 
	 * @param data a 1D {@code double} array
	 * 
	 * @param dim 1: column matrix; 2: row matrix
	 * 
	 */
	public DenseMatrix(double[] data, int dim) {
		if (dim == 1) {
			M = data.length;
			N = 1;
			this.data = new double[M][];
			for (int i = 0; i < M; i++) {
				this.data[i] = new double[N];
				this.data[i][0] = data[i];
			}
		} else if (dim == 2) {
			M = 1;
			N = data.length;
			this.data = new double[M][];
			this.data[0] = data;
		}
	}
	
	/**
	 * Construct an M x N dense matrix and initialize all elements 
	 * by a real number v.
	 * 
	 * @param M number of rows
	 * 
	 * @param N number of columns
	 * 
	 * @param v a real number
	 */
	public DenseMatrix(int M, int N, double v) {
		data = new double[M][];
		for (int i = 0; i < M; i++) {
			data[i] = new double[N];
			for (int j = 0; j < N; j++) {
				data[i][j] = v;
			}
		}
		this.M = M;
		this.N = N;
	}
	
	/**
	 * Construct a dense matrix with a specified size and
	 * initialize all elements by a real number v.
	 * 
	 * @param size size[0] is the number of rows and size[1] is 
	 *             the number of columns
	 *             
	 * @param v a real number
	 */
	public DenseMatrix(int[] size, double v) {
		if (size.length != 2) {
			System.err.println("The input integer array should have exactly two entries!");
			System.exit(1);
		}
		int M = size[0];
		int N = size[1];
		data = new double[M][];
		for (int i = 0; i < M; i++) {
			data[i] = new double[N];
			for (int j = 0; j < N; j++) {
				data[i][j] = v;
			}
		}
		this.M = M;
		this.N = N;
	}

	public double[][] getData() {
		return data;
	}
	
	public int getRowDimension() {
		return M;
	}

	public int getColumnDimension() {
		return N;
	}

	/**
	 * This * A. Since this is a dense matrix, the result must be
	 * a dense matrix.
	 */
	public Matrix mtimes(Matrix A) {
		
		Matrix res = null;
		
		double[][] resData = new double[M][];
		int NA = A.getColumnDimension();
		for (int i = 0; i < M; i++) {
			resData[i] = new double[NA];
		}
		
		double[] rowData = null;
		
		if (A instanceof DenseMatrix) {
			
			double[][] AData = ((DenseMatrix) A).getData();
			double[] columnA = new double[A.getRowDimension()];
			double s = 0;
			for (int j = 0; j < NA; j++) {
				for (int r = 0; r < A.getRowDimension(); r++) {
					columnA[r] = AData[r][j];
				}
				for (int i = 0; i < M; i++) {
					rowData = data[i];
					s = 0;
					for (int k = 0; k < N; k++) {
						// Using AData[k][j] costs 16.8 seconds
						// Referring AData[k][j] involves one integer multiplication!
						// s += rowData[k] * AData[k][j];
						// Using columnA[j] costs 3.4 seconds
						s += rowData[k] * columnA[k];
					}
					resData[i][j] = s;
				}
			}
			
		} else if (A instanceof SparseMatrix) {
				
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			int r = -1;
			double s = 0;

			for (int i = 0; i < M; i++) {
				rowData = data[i];
				for (int j = 0; j < NA; j++) {
					s = 0;
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						r = ir[k];
						// A[r][j] = pr[k]
						s += rowData[r] * pr[k];
					}
					resData[i][j] = s;
				}
			}

		}
		
		res = new DenseMatrix(resData);
		return res;
		
	}

	public double getEntry(int r, int c) {
		return data[r][c];
	}

	public void setEntry(int r, int c, double v) {
		data[r][c] = v;
	}

	public Matrix transpose() {
		
		double[][] resData = new double[N][];
		for (int i = 0; i < N; i++) {
			resData[i] = new double[M];
			for (int j = 0; j < M; j++) {
				resData[i][j] = data[j][i];
			}
		}
		
		return new DenseMatrix(resData);
		
	}

	public Matrix plus(Matrix A) {
		
		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		DenseMatrix res = (DenseMatrix) this.copy();
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double[] row = null;
			for (int i = 0; i < M; i++) {
				row = res.data[i];
				ARow = AData[i];
				for (int j = 0; j < N; j++) {
					row[j] += ARow[j];
				}
			}
		} else if (A instanceof SparseMatrix) {
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			int r = -1;

			for (int j = 0; j < A.getColumnDimension(); j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					r = ir[k];
					// A[r][j] = pr[k]
					res.data[r][j] += pr[k];
				}
			}
		}
		
		return res;
		
	}

	public Matrix minus(Matrix A) {

		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		DenseMatrix res = (DenseMatrix) this.copy();
		
		if (A instanceof DenseMatrix) {
			
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					res.data[i][j] -= ((DenseMatrix) A).data[i][j];
				}
			}
			
		} else if (A instanceof SparseMatrix) {
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			int r = -1;

			for (int j = 0; j < A.getColumnDimension(); j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					r = ir[k];
					// A[r][j] = pr[k]
					res.data[r][j] -= pr[k];
				}
			}
		}
		
		return res;
		
	}

	public Matrix times(Matrix A) {

		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		// double[][] resData = ((DenseMatrix) this.copy()).getData();
		double[][] resData = ArrayOperator.allocate2DArray(M, N, 0);
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double[] thisRow = null;
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				thisRow = data[i];
				ARow = AData[i];
				resRow = resData[i];
				thisRow = data[i];
				for (int j = 0; j < N; j++) {
					resRow[j] = thisRow[j] * ARow[j];
				}
			}
		} else if (A instanceof SparseMatrix) {
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			int r = -1;

			for (int j = 0; j < A.getColumnDimension(); j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					r = ir[k];
					// A[r][j] = pr[k]
					resData[r][j] = data[r][j] * pr[k];
				}
			}
		}
		
		return new DenseMatrix(resData);
		
	}

	public Matrix times(double v) {
		DenseMatrix res = (DenseMatrix) this.copy();
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				res.data[i][j] *= v;
			}
		}
		return res;
	}

	public Matrix copy() {
		DenseMatrix res = new DenseMatrix();
		res.M = M;
		res.N = N;
		res.data = new double[M][];
		for (int i = 0; i < M; i++) {
			res.data[i] = data[i].clone();
		}
		return res;
	}
	
	@Override
	public Matrix clone() {
		return this.copy();
	}

	public Matrix plus(double v) {
		DenseMatrix res = (DenseMatrix) this.copy();
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				res.data[i][j] += v;
			}
		}
		return res;
	}

	public Matrix minus(double v) {
		DenseMatrix res = (DenseMatrix) this.copy();
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				res.data[i][j] -= v;
			}
		}
		return res;
	}

	public Vector operate(Vector b) {
		
		if (N != b.getDim()) {
			System.err.println("Dimension does not match.");
			System.exit(1);
		}
		double[] V = new double[M];
		
		if (b instanceof DenseVector) {
			ArrayOperator.operate(V, data, ((DenseVector) b).getPr());
		} else if (b instanceof SparseVector) {
			int[] ir = ((SparseVector) b).getIr();
			double[] pr = ((SparseVector) b).getPr();
			int nnz = ((SparseVector) b).getNNZ();
			int idx = 0;
			double[] row_i = null;
			for (int i = 0; i < M; i++) {
				row_i = data[i];
				double s = 0;
				for (int k = 0; k < nnz; k++) {
					idx = ir[k];
					s += row_i[idx] * pr[k];
				}
				V[i] = s;
			}
		}
		
		return new DenseVector(V);
		
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(100);
		if (data == null) {
			sb.append("Empty matrix." + System.lineSeparator());
			return sb.toString();
		}
		int p = 4;
		for (int i = 0; i < getRowDimension(); i++) {
			sb.append("  ");
			for (int j = 0; j < getColumnDimension(); j++) {
				String valueString = "";
				double v = getEntry(i, j);
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				sb.append(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
				sb.append("  ");
			}
			sb.append(System.lineSeparator());
		}
		return sb.toString();
	}

	public void clear() {
		ArrayOperator.clearMatrix(data);
	}

	public Matrix getSubMatrix(int startRow, int endRow, int startColumn,
			int endColumn) {
		int nRow = endRow - startRow + 1;
		int nCol = endColumn - startColumn + 1;
		double[][] resData = new double[nRow][];
		double[] resRow = null;
		double[] thisRow = null;
		for (int r = 0, i = startRow; r < nRow; r++, i++) {
			resRow = new double[nCol];
			thisRow = data[i];
			System.arraycopy(thisRow, startColumn, resRow, 0, nCol);
			resData[r] = resRow;
		}
		return new DenseMatrix(resData);
	}

	public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns) {
		int nRow = selectedRows.length;
		int nCol = selectedColumns.length;
		double[][] resData = new double[nRow][];
		double[] resRow = null;
		double[] thisRow = null;
		for (int r = 0; r < nRow; r++) {
			resRow = new double[nCol];
			thisRow = data[selectedRows[r]];
			for (int c = 0; c < nCol; c++) {
				resRow[c] = thisRow[selectedColumns[c]];
			}
			resData[r] = resRow;
		}
		return new DenseMatrix(resData);
	}

	public Matrix getColumnMatrix(int c) {
		DenseMatrix res = new DenseMatrix(M, 1);
		double[][] resData = res.data;
		for (int i = 0; i < M; i++) {
			resData[i][0] = data[i][c];
		}
		return res;
	}

	public Vector getColumnVector(int c) {
		DenseVector res = new DenseVector(M);
		double[] pr = res.getPr();
		for (int i = 0; i < M; i++) {
			pr[i] = data[i][c];
		}
		return res;
	}

	public Matrix getRowMatrix(int r) {
		return new DenseMatrix(data[r], 2);
	}

	public Vector getRowVector(int r) {
		return new DenseVector(data[r]);
	}

	public Matrix getRows(int startRow, int endRow) {
		int numRows = endRow - startRow + 1;
		double[][] resData = new double[numRows][];
		for (int r = startRow, i = 0; r <= endRow; r++, i++) {
			resData[i] = data[r].clone();
		}
		return new DenseMatrix(resData);
	}

	public Matrix getRows(int... selectedRows) {
		int numRows = selectedRows.length;
		double[][] resData = new double[numRows][];
		for (int i = 0; i < numRows; i++) {
			resData[i] = data[selectedRows[i]].clone();
		}
		return new DenseMatrix(resData);
	}

	public Vector[] getRowVectors(int startRow, int endRow) {
		int numRows = endRow - startRow + 1;
		Vector[] res = new DenseVector[numRows];
		for (int r = startRow, i = 0; r <= endRow; r++, i++) {
			res[i] = new DenseVector(data[r]);
		}
		return res;
	}

	public Vector[] getRowVectors(int... selectedRows) {
		int numRows = selectedRows.length;
		Vector[] res = new DenseVector[numRows];
		for (int i = 0; i < numRows; i++) {
			res[i] = new DenseVector(data[selectedRows[i]]);
		}
		return res;
	}

	public Matrix getColumns(int startColumn, int endColumn) {
		int nRow = M;
		int nCol = endColumn - startColumn + 1;
		double[][] resData = new double[nRow][];
		double[] resRow = null;
		double[] thisRow = null;
		for (int r = 0; r < nRow; r++) {
			resRow = new double[nCol];
			thisRow = data[r];
			System.arraycopy(thisRow, startColumn, resRow, 0, nCol);
			resData[r] = resRow;
		}
		return new DenseMatrix(resData);
	}

	public Matrix getColumns(int... selectedColumns) {
		int nRow = M;
		int nCol = selectedColumns.length;
		double[][] resData = new double[nRow][];
		double[] resRow = null;
		double[] thisRow = null;
		for (int r = 0; r < nRow; r++) {
			resRow = new double[nCol];
			thisRow = data[r];
			for (int c = 0; c < nCol; c++) {
				resRow[c] = thisRow[selectedColumns[c]];
			}
			resData[r] = resRow;
		}
		return new DenseMatrix(resData);
	}

	public Vector[] getColumnVectors(int startColumn, int endColumn) {
		int numColumns = endColumn - startColumn + 1;
		Vector[] res = new DenseVector[numColumns];
		for (int c = startColumn, i = 0; c <= endColumn; c++, i++) {
			res[i] = getColumnVector(c);
		}
		return res;
	}

	public Vector[] getColumnVectors(int... selectedColumns) {
		int numColumns = selectedColumns.length;
		Vector[] res = new DenseVector[numColumns];
		for (int j = 0; j < numColumns; j++) {
			res[j] = getColumnVector(selectedColumns[j]);
		}
		return res;
	}

	public void setRowMatrix(int r, Matrix A) {
		if (A.getRowDimension() != 1) {
			err("Input matrix should be a row matrix.");
			exit(1);
		}
		double[] thisRow = data[r];
		if (A instanceof DenseMatrix) {
			double[] ARow = ((DenseMatrix) A).data[0];
			System.arraycopy(ARow, 0, thisRow, 0, N);
		} else if (A instanceof SparseMatrix) {
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			for (int j = 0; j < N; j++) {
				if (jc[j + 1] == jc[j]) {
					thisRow[j] = 0;
				} else {
					thisRow[j] = pr[jc[j]];
				}
			}
		}
	}

	public void setRowVector(int r, Vector V) {
		double[] thisRow = data[r];
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			System.arraycopy(pr, 0, thisRow, 0, N);
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int lastIdx = -1;
			int currentIdx = 0;
			for (int k = 0; k < nnz; k++) {
				currentIdx = ir[k];
				for (int j = lastIdx + 1; j < currentIdx; j++) {
					thisRow[j] = 0;
				}
				thisRow[currentIdx] = pr[k];
				lastIdx = currentIdx;
			}
			for (int j = lastIdx + 1; j < N; j++) {
				thisRow[j] = 0;
			}
		}
	}

	public void setColumnMatrix(int c, Matrix A) {
		if (A.getColumnDimension() != 1) {
			err("Input matrix should be a column matrix.");
			exit(1);
		}
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).data;
			for (int i = 0; i < M; i++) {
				data[i][c] = AData[i][0];
			}
		} else if (A instanceof SparseMatrix) {
			int[] jc = ((SparseMatrix) A).getJc();
			if (jc[1] == 0) {
				for (int i = 0; i < M; i++) {
					data[i][c] = 0;
				}
				return;
			}
			int[] ir = ((SparseMatrix) A).getIr();
			double[] pr = ((SparseMatrix) A).getPr();
			int lastIdx = -1;
			int currentIdx = 0;
			for (int k = 0; k < jc[1]; k++) {
				currentIdx = ir[k];
				for (int i = lastIdx + 1; i < currentIdx; i++) {
					data[i][c] = 0;
				}
				data[currentIdx][c] = pr[k];
				lastIdx = currentIdx;
			}
			for (int i = lastIdx + 1; i < M; i++) {
				data[i][c] = 0;
			}
		}
	}

	public void setColumnVector(int c, Vector V) {
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			for (int i = 0; i < M; i++) {
				data[i][c] = pr[i];
			}
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int lastIdx = -1;
			int currentIdx = 0;
			for (int k = 0; k < nnz; k++) {
				currentIdx = ir[k];
				for (int i = lastIdx + 1; i < currentIdx; i++) {
					data[i][c] = 0;
				}
				data[currentIdx][c] = pr[k];
				lastIdx = currentIdx;
			}
			for (int i = lastIdx + 1; i < M; i++) {
				data[i][c] = 0;
			}
		}
	}

}
