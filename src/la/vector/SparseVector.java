package la.vector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.utils.ArrayOperator;
import ml.utils.Pair;

/***
 * A Java implementation of sparse vectors.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 6th, 2013
 */
public class SparseVector implements Vector, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4760099385084043335L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	private int[] ir;
	
	private double[] pr;
	
	private int nnz;
	
	private int dim;

	public SparseVector(int dim) {
		ir = new int[0];
		pr = new double[0];
		nnz = 0;
		this.dim = dim;
	}
	
	public SparseVector(int[] ir, double[] pr, int nnz, int dim) {
		this.ir = ir;
		this.pr = pr;
		this.nnz = nnz;
		this.dim = dim;
	}
	
	/**
	 * Assign this sparse vector by a sparse vector V in the sense that
	 * all interior arrays of this vector are deep copy of the given
	 * sparse vector V.
	 * 
	 * @param V a sparse Vector
	 */
	public void assignSparseVector(SparseVector V) {
		ir = V.ir.clone();
		pr = V.pr.clone();
		nnz = V.nnz;
		dim = V.dim;
	}
	
	public int[] getIr() {
		return ir;
	}
	
	public double[] getPr() {
		return pr;
	}
	
	public int getNNZ() {
		return nnz;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer(100);
		sb.append('[');
		for (int k = 0; k < nnz; k++) {
			sb.append(String.format("%d: %.4f", ir[k], pr[k]));
			if (k < nnz - 1) {
				sb.append(", ");
			}
		}
		sb.append(']');
		return sb.toString();
	}

	@Override
	public int getDim() {
		return dim;
	}
	
	/**
	 * Change dimensionality of this sparse vector.
	 * 
	 * @param dim new dimensionality
	 */
	public void setDim(int dim) {
		if (dim > this.dim)
			this.dim = dim;
	}

	@Override
	public Vector copy() {
		return new SparseVector(ir.clone(), pr.clone(), nnz, dim);
	}
	
	@Override
	public Vector clone() {
		return this.copy();
	}

	@Override
	public Vector times(Vector V) {
		if (V.getDim() != this.dim) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			return V.times(this);
		} else if (V instanceof SparseVector) {
			ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz2 = ((SparseVector) V).getNNZ();
			if (this.nnz != 0 && nnz2 != 0) {
				int k1 = 0;
				int k2 = 0;
				int r1 = 0;
				int r2 = 0;
				double v = 0;
				int i = -1;
				while (k1 < this.nnz && k2 < nnz2) {
					r1 = this.ir[k1];
					r2 = ir[k2];
					if (r1 < r2) {
						k1++;
					} else if (r1 == r2) {
						i = r1;
						v = this.pr[k1] * pr[k2];
						k1++;
						k2++;
						if (v != 0) {
							list.add(Pair.of(i, v));
						}
					} else {
						k2++;
					}
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
		if (V.getDim() != this.dim) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			return V.plus(this);
		} else if (V instanceof SparseVector) {
			ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz2 = ((SparseVector) V).getNNZ();
			if (!(this.nnz == 0 && nnz2 == 0)) {
				int k1 = 0;
				int k2 = 0;
				int r1 = 0;
				int r2 = 0;
				double v = 0;
				int i = -1;
				while (k1 < this.nnz || k2 < nnz2) {
					if (k2 == nnz2) { // V has been processed.
						i = this.ir[k1];
						v = this.pr[k1];
						k1++;
					} else if (k1 == this.nnz) { // this has been processed.
						i = ir[k2];
						v = pr[k2];
						k2++;
					} else { // Both this and V have not been fully processed.
						r1 = this.ir[k1];
						r2 = ir[k2];
						if (r1 < r2) {
							i = r1;
							v = this.pr[k1];
							k1++;
						} else if (r1 == r2) {
							i = r1;
							v = this.pr[k1] + pr[k2];
							k1++;
							k2++;
						} else {
							i = r2;
							v = pr[k2];
							k2++;
						}
					}
					if (v != 0) {
						list.add(Pair.of(i, v));
					}
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
	public Vector minus(Vector V) {
		if (V.getDim() != this.dim) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (V instanceof DenseVector) {
			double[] VPr = ((DenseVector) V).getPr();
			double[] resPr = new double[dim];
			for (int k = 0; k < dim; k++) {
				resPr[k] = -VPr[k];
			}
			for (int k = 0; k < nnz; k++) {
				resPr[ir[k]] += pr[k];
			}
			return new DenseVector(resPr);
		} else if (V instanceof SparseVector) {
			ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz2 = ((SparseVector) V).getNNZ();
			if (!(this.nnz == 0 && nnz2 == 0)) {
				int k1 = 0;
				int k2 = 0;
				int r1 = 0;
				int r2 = 0;
				double v = 0;
				int i = -1;
				while (k1 < this.nnz || k2 < nnz2) {
					if (k2 == nnz2) { // V has been processed.
						i = this.ir[k1];
						v = this.pr[k1];
						k1++;
					} else if (k1 == this.nnz) { // this has been processed.
						i = ir[k2];
						v = -pr[k2];
						k2++;
					} else { // Both this and V have not been fully processed.
						r1 = this.ir[k1];
						r2 = ir[k2];
						if (r1 < r2) {
							i = r1;
							v = this.pr[k1];
							k1++;
						} else if (r1 == r2) {
							i = r1;
							v = this.pr[k1] - pr[k2];
							k1++;
							k2++;
						} else {
							i = r2;
							v = -pr[k2];
							k2++;
						}
					}
					if (v != 0) {
						list.add(Pair.of(i, v));
					}
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
	public double get(int i) {
		
		if (i < 0 || i >= dim) {
			System.err.println("Wrong index.");
			System.exit(1);
		}
		
		if (nnz == 0) {
			return 0;
		}
		
		int u = nnz - 1;
		int l = 0;
		int idx = -1;
		int k = 0;
		while (true) {
			if (l > u) {
				break;
			}
			k = (u + l) / 2;
			idx = ir[k];
			if (idx == i) { // Hits
				return pr[k];
			} else if (idx < i) {
				l = k + 1;
			} else {
				u = k - 1;
			}
		}
		
		return 0;
		
	}

	@Override
	public void set(int i, double v) {
		
		if (i < 0 || i >= dim) {
			System.err.println("Wrong index.");
			System.exit(1);
		}
		
		if (nnz == 0) {
			insertEntry(i, v, 0);
			return;
		}
		
		int u = nnz - 1;
		int l = 0;
		int idx = -1;
		int k = 0;
		
		int flag = 0;
		while (true) {
			if (l > u) {
				break;
			}
			k = (u + l) / 2;
			idx = ir[k];
			if (idx == i) { // Hits
				if (v == 0)
					deleteEntry(k);
				else
					pr[k] = v;
				return;
			} else if (idx < i) {
				l = k + 1;
				flag = 1;
			} else {
				u = k - 1;
				flag = 2;
			}
		}
		if (flag == 1) {
			k++;
		}
		insertEntry(i, v, k);
		
	}

	private void insertEntry(int r, double v, int pos) {
		
		if (v == 0) {
			return;
		}
		
		int len_old = pr.length;
		
		int new_space = len_old < dim - 10 ? 10 : dim - len_old;
		
		if (nnz + 1 > len_old) {
			double[] pr_new = new double[len_old + new_space];
			System.arraycopy(pr, 0, pr_new, 0, pos);
			pr_new[pos] = v;
			if (pos < len_old)
				System.arraycopy(pr, pos, pr_new, pos + 1, len_old - pos);
			pr = pr_new;
		} else {
			for (int i = nnz - 1; i >= pos; i--) {
				pr[i + 1] = pr[i];
			}
			pr[pos] = v;
		}
		
		if (nnz + 1 > len_old) {
			int[] ir_new = new int[len_old + new_space];
			System.arraycopy(ir, 0, ir_new, 0, pos);
			ir_new[pos] = r;
			if (pos < len_old)
				System.arraycopy(ir, pos, ir_new, pos + 1, len_old - pos);
			ir = ir_new;
		} else {
			for (int i = nnz - 1; i >= pos; i--) {
				ir[i + 1] = ir[i];
			}
			ir[pos] = r;
		}
		
		nnz++;
		
	}
	
	/**
	 * Clean entries so that zero entries are removed.
	 */
	public void clean() {
		for (int k = nnz - 1; k >= 0; k--) {
			if (pr[k] == 0) {
				deleteEntry(k);
			}
		}
	}

	private void deleteEntry(int pos) {

		// The pos-th entry in pr must exist

		for (int i = pos; i < nnz - 1; i++) {
			pr[i] = pr[i + 1];
			ir[i] = ir[i + 1];
		}
		
		nnz--;

	}

	@Override
	public Vector operate(Matrix A) {
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M != dim) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (A instanceof DenseMatrix) {
			double[] res = ArrayOperator.allocate1DArray(N, 0);
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double v = 0;
			for (int k = 0; k < nnz; k++) {
				int i = ir[k];
				ARow = AData[i];
				v = this.pr[k];
				for (int j = 0; j < N; j++) {
					res[j] += v * ARow[j];
				}
			}
			return new DenseVector(res);
		} else if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			double s = 0;
			int k1 = 0;
			int k2 = 0;
			int c = 0;
			int r = 0;
			TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
			for (int j = 0; j < N; j++) {
				k1 = 0;
				k2 = jc[j];
				s = 0;
				while (true) {
					if (k2 >= jc[j + 1] || k1 >= nnz) {
						break;
					}
					c = this.ir[k1];
					r = ir[k2];
					if (r < c) {
						k2++;
					} else if (r > c) {
						k1++;
					} else {
						s += this.pr[k1] * pr[k2];
						k1++;
						k2++;
					}
				}
				if (s != 0) {
					map.put(j, s);
				}	
			}
			int nnz = map.size();
			ir = new int[nnz];
			pr = new double[nnz];
			int ind = 0;
			for (Entry<Integer, Double> entry : map.entrySet()) {
				ir[ind] = entry.getKey();
				pr[ind] = entry.getValue();
				ind++;
			}
			return new SparseVector(ir, pr, nnz, N);
		}
		return null;
	}

	@Override
	public void clear() {
		ir = new int[0];
		pr = new double[0];
		nnz = 0;
	}

	@Override
	public Vector times(double v) {
		if (v == 0) {
			return new SparseVector(dim);
		}
		SparseVector res = (SparseVector) this.copy();
		double[] pr = res.pr;
		for (int k = 0; k < nnz; k++) {
			pr[k] *= v;
		}
		return res;
	}

}
