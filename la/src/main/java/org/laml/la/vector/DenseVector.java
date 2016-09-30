package org.laml.la.vector;

import java.io.Serializable;
import java.util.ArrayList;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.utils.ArrayOperator;
import org.laml.la.utils.Pair;

/***
 * A Java implementation of dense vectors.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 6th, 2013
 */
public class DenseVector implements Vector, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6411390717530519480L;
	private double[] pr;

	public DenseVector() {
	}
	
	public DenseVector(int dim, double v) {
		pr = ArrayOperator.allocateVector(dim, v);
	}
	
	public DenseVector(int dim) {
		pr = ArrayOperator.allocateVector(dim, 0);
	}
	
	public DenseVector(double[] pr) {
		this.pr = pr;
	}
	
	public static DenseVector buildDenseVector(double[] pr) {
		DenseVector res = new DenseVector();
		res.pr = pr;
		return res;
	}

	public double[] getPr() {
		return pr;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer(100);
		sb.append('[');
		for (int k = 0; k < pr.length; k++) {
			sb.append(String.format("%.4f", pr[k]));
			if (k < pr.length - 1) {
				sb.append(", ");
			}
		}
		sb.append(']');
		return sb.toString();
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}

	@Override
	public int getDim() {
		return pr.length;
	}

	@Override
	public Vector copy() {
		return new DenseVector(pr.clone());
	}
	
	@Override
	public Vector clone() {
		return this.copy();
	}

	@Override
	public Vector times(Vector V) {
		if (V.getDim() != this.pr.length) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			return new DenseVector(ArrayOperator.times(pr, ((DenseVector) V).getPr()));
		} else if (V instanceof SparseVector) {
			/*double[] res = ArrayOperation.allocateVector(getDim(), 0);
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int idx = -1;
			for (int k = 0; k < ((SparseVector) V).getNNZ(); k++) {
				idx = ir[k];
				res[ir[k]] = this.pr[idx] * pr[k];
			}
			return new DenseVector(res);*/
			ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int idx = -1;
			double v = 0;
			for (int k = 0; k < ((SparseVector) V).getNNZ(); k++) {
				idx = ir[k];
				v = this.pr[idx] * pr[k];
				if (v != 0) {
					list.add(Pair.of(idx, v));
				}
			}
			int nnz = list.size();
			int dim = this.getDim();
			int[] ir_res = new int[nnz];
			double[] pr_res = new double[nnz];
			int k = 0;
			for (Pair<Integer, Double> pair : list) {
				ir_res[k] = pair.first;
				pr_res[k] = pair.second;
				k++;
			}
			return new SparseVector(ir_res, pr_res, nnz, dim);
		}
		return null;
	}

	@Override
	public Vector plus(Vector V) {
		if (V.getDim() != this.pr.length) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			return new DenseVector(ArrayOperator.plus(pr, ((DenseVector) V).getPr()));
		} else if (V instanceof SparseVector) {
			double[] res = this.pr.clone();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int idx = -1;
			for (int k = 0; k < ((SparseVector) V).getNNZ(); k++) {
				idx = ir[k];
				res[ir[k]] = this.pr[idx] + pr[k];
			}
			return new DenseVector(res);
		}
		return null;
	}

	@Override
	public Vector minus(Vector V) {
		if (V.getDim() != this.pr.length) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			return new DenseVector(ArrayOperator.minus(pr, ((DenseVector) V).getPr()));
		} else if (V instanceof SparseVector) {
			double[] res = this.pr.clone();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int idx = -1;
			for (int k = 0; k < ((SparseVector) V).getNNZ(); k++) {
				idx = ir[k];
				res[ir[k]] = this.pr[idx] - pr[k];
			}
			return new DenseVector(res);
		}
		return null;
	}

	@Override
	public double get(int i) {
		return pr[i];
	}

	@Override
	public void set(int i, double v) {
		pr[i] = v;
	}

	@Override
	public Vector operate(Matrix A) {
		
		int dim = getDim();
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M != dim) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		double[] res = ArrayOperator.allocate1DArray(N, 0);
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double v = 0;
			for (int i = 0; i < M; i++) {
				ARow = AData[i];
				v = this.pr[i];
				for (int j = 0; j < N; j++) {
					res[j] += v * ARow[j];
				}
			}
		} else if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			for (int j = 0; j < N; j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					res[j] += this.pr[ir[k]] * pr[k];
				}
			}
		}
		return new DenseVector(res);
	}

	@Override
	public void clear() {
		ArrayOperator.clearVector(pr);
	}

	@Override
	public Vector times(double v) {
		if (v == 0) {
			return new DenseVector(getDim(), 0);
		}
		double[] resData = pr.clone();
		for (int i = 0; i < pr.length; i++) {
			resData[i] *= v;
		}
		return new DenseVector(resData);
	}

}
