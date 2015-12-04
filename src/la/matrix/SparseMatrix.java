package la.matrix;

import static ml.utils.Printer.err;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Printer.sprintf;
import static ml.utils.Utility.exit;

import java.io.Serializable;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.TreeSet;

import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.utils.ArrayOperator;
import ml.utils.Pair;

/***
 * A Java implementation of sparse matrices with combined compressed 
 * sparse column (CSC) and compressed sparse row (CSR). The advantage
 * is that efficient operations can be obtained either by columns or
 * by rows.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 6th, 2013
 */
public class SparseMatrix implements Matrix, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 404718895052720649L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Matrix S = new SparseMatrix(3, 3);
		printMatrix(S);
		
		Matrix A = new DenseMatrix(3, 3);
		printMatrix(S.mtimes(A));
		printMatrix(A.mtimes(S));

	}

	
	private SparseMatrix() {
		M = 0;
		N = 0;
		nzmax = 0;
		nnz = 0;
	}
	
	/**
	 * Constructor for an M x N sparse matrix.
	 * 
	 * @param M number of rows
	 * 
	 * @param N number of columns
	 */
	public SparseMatrix(int M, int N) {
		this.M = M;
		this.N = N;
		this.nzmax = 0;
		this.nnz = 0;
		this.jc = new int[N + 1];
		for (int j = 0; j < N + 1; j++) {
			jc[j] = 0;
		}
		this.jr = new int[M + 1];
		for (int i = 0; i < M + 1; i++) {
			jr[i] = 0;
		}
		ir = new int[0];
		pr = new double[0];
		ic = new int[0];
		valCSRIndices = new int[0];
	}
	
	public SparseMatrix(SparseMatrix A) {
		ir = A.ir;
		jc = A.jc;
		pr = A.pr;
		ic = A.ic;
		jr = A.jr;
		valCSRIndices = A.valCSRIndices;
		M = A.M;
		N = A.N;
		nzmax = A.nzmax;
		nnz = A.nnz;
	}
	
	public SparseMatrix(int[] rIndices, int[] cIndices, double[] values, int numRows, int numColumns, int nzmax) {
		SparseMatrix temp = createSparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		assignSparseMatrix(temp);
	}
	
	/**
	 * Assign this sparse matrix by a sparse matrix A in the sense that
	 * all interior arrays of this matrix are deep copy of the given
	 * sparse matrix A.
	 * 
	 * @param A a sparse Matrix
	 */
	public void assignSparseMatrix(SparseMatrix A) {
		ir = A.ir.clone();
		jc = A.jc.clone();
		pr = A.pr.clone();
		ic = A.ic.clone();
		jr = A.jr.clone();
		valCSRIndices = A.valCSRIndices.clone();
		M = A.M;
		N = A.N;
		nzmax = A.nzmax;
		nnz = A.nnz;
	}
	
	/**
	 * Create a sparse matrix from a map from index pairs to values.
	 * 
	 * @param inputMap a mapping from row-column pair to value
	 * 
	 * @param numRows number of rows
	 * 
	 * @param numColumns number of columns
	 * 
	 * @return a sparse matrix
	 */
	public static SparseMatrix createSparseMatrix(TreeMap<Pair<Integer, Integer>, Double> inputMap, int numRows, int numColumns) {
		
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		int nzmax = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : inputMap.entrySet()) {
			if (entry.getValue() != 0) {
				map.put(Pair.of(entry.getKey().second, entry.getKey().first), entry.getValue());
				nzmax++;
			}
		}
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];
		
		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;
		
		return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}
	
	/**
	 * Create a sparse matrix from index and value arrays.
	 * 
	 * @param rIndices a 1D {@code int} array of row indices for 
	 *           	   non-zero elements
	 *           
	 * @param cIndices a 1D {@code int} array of column indices for 
	 *                 non-zero elements
	 *                 
	 * @param values a 1D {@code double} array of non-zero elements
	 * 
	 * @param numRows number of rows
	 * 
	 * @param numColumns number of columns
	 * 
	 * @param nzmax maximal number of non-zero elements
	 * 
	 * @return a sparse matrix
	 */
	public static SparseMatrix createSparseMatrix(int[] rIndices, int[] cIndices, double[] values, int numRows, int numColumns, int nzmax) {
		
		int k = -1;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (k = 0; k < values.length; k++) {
			if (values[k] == 0) {
				continue;
			}
			// Order by column then row
			map.put(Pair.of(cIndices[k], rIndices[k]), values[k]);
		}
		
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];
		
		int rIdx = -1;
		int cIdx = -1;
		k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;
		
		return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}
	
	/**
	 * Create a sparse matrix from index and value arrays.
	 * 
	 * @param rIndices a 1D {@code int} array of row indices for 
	 *           	   non-zero elements
	 *           
	 * @param cIndices a 1D {@code int} array of column indices for 
	 *                 non-zero elements
	 *                 
	 * @param values a 1D {@code double} array of non-zero elements
	 * 
	 * @param numRows number of rows
	 * 
	 * @param numColumns number of columns
	 */
	public SparseMatrix(int[] rIndices, int[] cIndices, double[] values, int numRows, int numColumns) {
		this(rIndices, cIndices, values, numRows, numColumns, values.length);
	}
	
	/**
	 * Create a sparse matrix with the compressed sparse column format.
	 * 
	 * @param ir a 1D {@code int} array of row indices for 
	 *           non-zero elements
	 * 
	 * @param jc a 1D {@code int} array of number of non-zero 
	 *           elements in each column
	 *           
	 * @param pr a 1D {@code double} array of non-zero elements
	 * 
	 * @param M number of rows
	 * 
	 * @param N number of columns
	 * 
	 * @param nzmax maximal number of non-zero entries
	 * 
	 * @return a sparse matrix
	 */
	public static SparseMatrix createSparseMatrixByCSCArrays(int[] ir, int[] jc, double[] pr, int M, int N, int nzmax) {
		
		SparseMatrix res = new SparseMatrix();
		res.ir = ir;
		res.jc = jc;
		res.pr = pr;
		res.M = M;
		res.N = N;
		res.nzmax = nzmax;
		
		int[] ic = new int[pr.length];
		int[] jr = new int[M + 1];
		int[] valCSRIndices = new int[pr.length];
		
		/*
		 * Transform compressed sparse column format to 
		 * compressed sparse row format.
		 */
		int[] rIndices = ir;
		int[] cIndices = new int[ic.length];
		int k = 0;
		int j = 0;
		while (k < ir.length && j < N) {
			if (jc[j] <= k && k < jc[j + 1]) {
				cIndices[k] = j;
				k++;
			} else {
				j++;
			}
		}
		
		TreeMap<Pair<Integer, Integer>, Integer> map = new TreeMap<Pair<Integer, Integer>, Integer>();
		for (k = 0; k < pr.length; k++) {
			if (pr[k] == 0) {
				continue;
			}
			// Order by row then column
			map.put(Pair.of(rIndices[k], cIndices[k]), k);
		}
		
		int rIdx = -1;
		int cIdx = -1;
		int vIdx = -1;
		k = 0;
		jr[0] = 0;
		int currentRow = 0;
		for (Entry<Pair<Integer, Integer>, Integer> entry : map.entrySet()) {
			rIdx = entry.getKey().first;
			cIdx = entry.getKey().second;
			vIdx = entry.getValue();
			ic[k] = cIdx;
			valCSRIndices[k] = vIdx;
			while (currentRow < rIdx) {
				jr[currentRow + 1] = k;
				currentRow++;
			}
			k++;
		}
		while (currentRow < M) {
			jr[currentRow + 1] = k;
			currentRow++;
		}
		jr[M] = k;
		
		res.ic = ic;
		res.jr = jr;
		res.valCSRIndices = valCSRIndices;
		res.nnz = map.size();
		
		return res;
		
	}

	/**
	 * Create a sparse matrix with the compressed sparse column format
	 * using compressed sparse row information.
	 * 
	 * @param ic a 1D {@code int} array of column indices for 
	 *           non-zero elements
	 * 
	 * @param jr a 1D {@code int} array of number of non-zero 
	 *           elements in each row
	 * 
	 * @param pr a 1D {@code double} array of non-zero elements
	 * 
	 * @param M number of rows
	 * 
	 * @param N number of columns
	 * 
	 * @param nzmax maximal number of non-zero entries
	 * 
	 * @return a sparse matrix
	 */
	public static SparseMatrix createSparseMatrixByCSRArrays(int[] ic, int[] jr, double[] pr, int M, int N, int nzmax) {
		
		/*
		 * Transform compressed sparse row format to 
		 * compressed sparse column format.
		 */
		int[] rIndices = new int[ic.length];
		int[] cIndices = ic;
		int k = 0;
		int i = 0;
		while (k < ic.length && i < M) {
			if (jr[i] <= k && k < jr[i + 1]) {
				rIndices[k] = i;
				k++;
			} else {
				i++;
			}
		}
		
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (k = 0; k < pr.length; k++) {
			if (pr[k] == 0) {
				continue;
			}
			// Order by column then row
			map.put(Pair.of(cIndices[k], rIndices[k]), pr[k]);
		}

		int numRows = M;
		int numColumns = N;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);

	}

	/**
	 * Row index array.
	 */
	private int[] ir;
	
	/**
	 * First entry index for each column.
	 */
	private int[] jc;
	
	/**
	 * Column index array.
	 */
	private int[] ic;
	
	/**
	 * First entry index for each row.
	 */
	private int[] jr;
	
	/**
	 * Pointer to the non-zero value array.
	 */
	private double[] pr;
	
	/**
	 * Ordered value indices when converting CSC to CSR, i.e., 
	 * the k-th non-zero value of a CSR sparse matrix is the 
	 * valCSRIndices[k]-th non-zero value of a CSC sparse matrix.
	 */
	private int[] valCSRIndices;
	
	/**
	 * Number of non-zero entries in this sparse matrix.
	 */
	private int nnz;
	
	/**
	 * Maximal number of non-zero entries in this sparse matrix.
	 */
	private int nzmax;
	
	/**
	 * Number of rows.
	 */
	private int M;
	
	/**
	 * Number of columns.
	 */
	private int N;

	public int[] getIr() {
		return ir;
	}
	
	public int[] getJc() {
		return jc;
	}
	
	public int[] getIc() {
		return ic;
	}
	
	public int[] getJr() {
		return jr;
	}
	
	public double[] getPr() {
		return pr;
	}
	
	public int[] getValCSRIndices() {
		return valCSRIndices;
	}
	
	@Override
	public int getRowDimension() {
		return M;
	}

	@Override
	public int getColumnDimension() {
		return N;
	}
	
	public int getNZMax() {
		return nzmax;
	}
	
	public int getNNZ() {
		return nnz;
	}
	
	@Override
	public double[][] getData() {
		double[][] data = new double[M][];
		for (int i = 0; i < M; i++) {
			double[] rowData = ArrayOperator.allocateVector(N, 0);
			for (int k = jr[i]; k < jr[i + 1]; k++) {
				rowData[ic[k]] = pr[valCSRIndices[k]];
			}
			data[i] = rowData;
		}
		return data;
	}

	@Override
	/**
	 * This * A.
	 */
	public Matrix mtimes(Matrix A) {
		
		Matrix res = null;
		int NA = A.getColumnDimension();
		
		if (A instanceof DenseMatrix) {
			
			double[][] resData = new double[M][];
			for (int i = 0; i < M; i++) {
				resData[i] = new double[NA];
			}
			double[] resRow = null;
			double[][] data = ((DenseMatrix) A).getData();
			
			int c = -1;
			double s = 0;
			// double v = 0;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < NA; j++) {
					s = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						c = ic[k];
						s += pr[valCSRIndices[k]] * data[c][j];
					}
					resRow[j] = s;
				}
			}
			
			res = new DenseMatrix(resData);

		} else if (A instanceof SparseMatrix) {

			/*
			 * When this and A are all sparse matrices,
			 * the result is also a sparse matrix.
			 */
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			// rowIdx of the right sparse matrix
			int rr = -1;
			// colIdx of the left sparse matrix
			int cl = -1;
			double s = 0;
			int kl = 0;
			int kr = 0;
			int nzmax = 0;
			
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

			for (int i = 0; i < M; i++) {
				for (int j = 0; j < NA; j++) {
					s = 0;
					kl = jr[i];
					kr = jc[j];
					while (true) {
						if (kl >= jr[i + 1] || kr >= jc[j + 1]) {
							break;
						}
						cl = ic[kl];
						rr = ir[kr];
						if (cl < rr) {
							kl++;
						} else if (cl > rr) {
							kr++;
						} else {
							s += this.pr[valCSRIndices[kl]] * pr[kr];
							kl++;
							kr++;
						}
					}
					
					// Order by column then row
					if (s != 0) {
						nzmax++;
						map.put(Pair.of(j, i), s);
					}	
				}
			}
			
			int numRows = this.M;
			int numColumns = NA;
			ir = new int[nzmax];
			jc = new int[numColumns + 1];
			pr = new double[nzmax];
			
			int rIdx = -1;
			int cIdx = -1;
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;
			
			res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
			
		}
		
		return res;
		
	}

	@Override
	public double getEntry(int r, int c) {
		if (r < 0 || r > M - 1 || c < 0 || c > N - 1) {
			System.err.println("Wrong index.");
			System.exit(1);
		}
		double res = 0;
		int idx = -1;
		if (r <= c) {
			// Retrieve the entry by the c-th column
			int u = jc[c + 1] - 1;
			int l = jc[c];
			if (u < l) { // The c-th column is empty
				return 0;
			}
			int k = jc[c];
			while (true) {
				if (l > u) {
					break;
				}
				k = (u + l) / 2;
				idx = ir[k];
				if (idx == r) { // Hits
					return pr[k];
				} else if (idx < r) {
					l = k + 1;
				} else {
					u = k - 1;
				}
			}
			/*for (int k = jc[c]; k < jc[c + 1]; k++) {
				idx = ir[k];
				if (idx == r) {
					res = pr[k];
					break;
				} else if (idx > r){
					break;
				}
			}*/
		} else {
			// Retrieve the entry by the r-th row
			int u = jr[r + 1] - 1;
			int l = jr[r];
			if (u < l) { // The c-th column is empty
				return 0;
			}
			int k = jr[r];
			while (true) {
				if (l > u) {
					break;
				}
				k = (u + l) / 2;
				idx = ic[k];
				if (idx == c) { // Hits
					return pr[valCSRIndices[k]];
				} else if (idx < c) {
					l = k + 1;
				} else {
					u = k - 1;
				}
			}
			/*for (int k = jr[r]; k < jr[r + 1]; k++) {
				idx = ic[k];
				if (idx == c) {
					res = pr[valCSRIndices[k]];
					break;
				} else if (idx > c){
					break;
				}
			}*/
		}
		return res;
	}

	@Override
	public void setEntry(int r, int c, double v) {
		if (r < 0 || r > M - 1 || c < 0 || c > N - 1) {
			System.err.println("Wrong index.");
			System.exit(1);
		}
		int u = jc[c + 1] - 1;
		int l = jc[c];
		if (u < l) { // The c-th column is empty
			insertEntry(r, c, v, jc[c]);
			return;
		}
		int idx = -1;
		/* If jc[c] == jc[c + 1] (meaning that the c-th column is empty), 
		 * jc[c] will be the insertion position
		 */
		int k = jc[c];
		int flag = 0;
		while (true) {
			if (l > u) {
				break;
			}
			k = (u + l) / 2;
			idx = ir[k];
			if (idx == r) { // Hits
				if (v == 0)
					deleteEntry(r, c, k);
				else
					pr[k] = v;
				return;
			} else if (idx < r) {
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
		insertEntry(r, c, v, k);
		
	}
	
	/**
	 * Insert a new entry (r, c) = v and its insertion position is pos in pr.
	 * @param r
	 * @param c
	 * @param v
	 * @param pos insertion position in pr, i.e., the new entry's 
	 *            position in pr
	 */
	private void insertEntry(int r, int c, double v, int pos) {
		
		if (v == 0) {
			return;
		}
		
		int len_old = pr.length;
		
		int new_space = len_old < M * N - 10 ? 10 : M * N - len_old;
		
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
		
		for (int j = c + 1; j < N + 1; j++) {
			jc[j]++;
		}
		
		int u = jr[r + 1] - 1; // u is the index of the last entry in r-th row
		int l = jr[r];		   // l is the index of the first entry in r-th row
		/* If jr[r] == jr[r + 1] (meaning that the r-th row is empty), 
		 * jr[r] will be the insertion position
		 */
		int k = jr[r];
		int idx = -1;
		int flag = 0;
		while (true) { // Assume there exists a k such that ic[k] = c.
			if (l > u) { // If l > u in the first round, the r-th row is empty,
				break;   // and k = jr[r] is the insertion position.
			}
			k = (u + l) / 2;
			idx = ic[k]; // Until now ic[l] <= c <= ic[u] and l <= u (0)
			if (idx == c) { // Hits (Actually, it won't hit)

			} else if (idx < c) {
				l = k + 1; // Assume there exists a k such that ic[k] = c,
				flag = 1;  // then ic[l] <= c (1).
			} else {
				u = k - 1; // Assume there exists a k such that ic[k] = c,
				flag = 2;  // then ic[u] >= c (2).
			}
			// Assume there exists a k such that ic[k] = c, then l <= u.
			// If l > u, it means that there doesn't exist a k such that ic[k] = c
			// Since l <= u before updating by (1) and (2), if l > u after either 
			// (1) or (2), we must have that only one of (1) or (2) breaks (0).
			// If (1) breaks (0), we have ic[l] > c and 
			// l = k + 1 is the insertion position.
			// If (2) breaks (0), we have ic[u] < c and 
			// u + 1 = k is the insertion position.
		}
		if (flag == 1) {
			k++;
		}
		
		if (nnz + 1 > len_old) {
			int[] ic_new = new int[len_old + new_space];
			System.arraycopy(ic, 0, ic_new, 0, k);
			ic_new[k] = c;
			if (k < len_old)
				System.arraycopy(ic, k, ic_new, k + 1, len_old - k);
			ic = ic_new;
		} else {
			for (int i = nnz - 1; i >= k; i--) {
				ic[i + 1] = ic[i];
			}
			ic[k] = c;
		}
		
		for (int i = r + 1; i < M + 1; i++) {
			jr[i]++;
		}

		// for (int i = 0; i < valCSRIndices.length; i++) {
		for (int i = 0; i < nnz; i++) {
			if (valCSRIndices[i] >= pos)
				valCSRIndices[i]++;
		}
		
		if (nnz + 1 > len_old) {
			int[] valCSRIndices_new = new int[len_old + new_space];
			System.arraycopy(valCSRIndices, 0, valCSRIndices_new, 0, k);
			valCSRIndices_new[k] = pos;
			if (k < len_old)
				System.arraycopy(valCSRIndices, k, valCSRIndices_new, k + 1, len_old - k);
			valCSRIndices = valCSRIndices_new;
		} else {
			for (int i = nnz - 1; i >= k; i--) {
				valCSRIndices[i + 1] = valCSRIndices[i];
			}
			valCSRIndices[k] = pos;
		}
		
		nnz++;
		if (nnz > len_old) {
			nzmax = len_old + new_space;
		}
		
	}
	
	/**
	 * Delete the entry indexed by (r, c) whose index of pr is pos.
	 * 
	 * @param r row index of the entry to be deleted
	 * 
	 * @param c column index of the entry to be deleted
	 * 
	 * @param pos index in pr of the (r, c) entry
	 */
	private void deleteEntry(int r, int c, int pos) {
		
		// The pos-th entry in pr must exist
		
		for (int i = pos; i < nnz - 1; i++) {
			pr[i] = pr[i + 1];
			ir[i] = ir[i + 1];
		}
		
		for (int j = c + 1; j < N + 1; j++) {
			jc[j]--;
		}
		
		int u = jr[r + 1] - 1;
		int l = jr[r];
		/* If jr[r] == jr[r + 1] (meaning that the r-th row is empty), 
		 * jr[r] will be the insertion position
		 */
		int k = jr[r]; 
		int idx = -1;
		// int flag = 0; // flag = 1: ic[k] < c; flag = 2: ic[k] > c
		while (true) {
			if (l > u) {
				break;
			}
			k = (u + l) / 2;
			idx = ic[k];
			if (idx == c) { // Hits
				// flag = 0;
				break;
			} else if (idx < c) {
				l = k + 1;
				// flag = 1;
			} else {
				u = k - 1;
				// flag = 2;
			}
		}
		/*if (flag == 1) {
			k++;
		}*/

		for (int i = 0; i < valCSRIndices.length; i++) {
			if (valCSRIndices[i] > pos)
				valCSRIndices[i]--;
		}
		
		for (int j = k; j < nnz - 1; j++) {
			ic[j] = ic[j + 1];
			valCSRIndices[j] = valCSRIndices[j + 1];
		}
		
		for (int i = r + 1; i < M + 1; i++) {
			jr[i]--;
		}
		
		nnz--;
		
	}

	@SuppressWarnings("unused")
	@Deprecated
	private Matrix transpose0() {
		
		double[] values = pr;
		int[] rIndices = ir;
		int[] cIndices = new int[ic.length];
		int k = 0;
		int j = 0;
		while (k < nnz && j < N) {
			if (jc[j] <= k && k < jc[j + 1]) {
				cIndices[k] = j;
				k++;
			} else {
				j++;
			}
		}
		
		return createSparseMatrix(cIndices, rIndices, values, N, M, nzmax);
		
	}
	
	@Override
	public Matrix transpose() {
		
		SparseMatrix res = new SparseMatrix();
		res.M = N;
		res.N = M;
		res.nnz = nnz;
		res.nzmax = nzmax;
		res.ir = ic.clone();
		res.jc = jr.clone();
		res.ic = ir.clone();
		res.jr = jc.clone();
		double[] pr_new = new double[nzmax];
		int k = 0;
		for (k = 0; k < nnz; k++) {
			pr_new[k] = pr[valCSRIndices[k]];
		}
		res.pr = pr_new;
		
		int[] valCSRIndices_new = new int[nzmax];
		int j = 0;
		int rIdx = -1;
		int cIdx = -1;
		k = 0;
		int k2 = 0;
		int numBeforeThisEntry = 0;
		while (k < nnz && j < N) {
			if (jc[j] <= k && k < jc[j + 1]) {
				rIdx = ir[k];
				cIdx = j;
				numBeforeThisEntry = jr[rIdx];
				for (k2 = jr[rIdx]; k2 < jr[rIdx + 1]; k2++) {
					if (ic[k2] == cIdx) {
						break;
					} else {
						numBeforeThisEntry++;
					}
				}
				valCSRIndices_new[k] = numBeforeThisEntry;
				k++;
			} else {
				j++;
			}
		}
		res.valCSRIndices = valCSRIndices_new;
		
		return res;
		
	}

	@Override
	public Matrix plus(Matrix A) {
		
		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		Matrix res = null;
		
		if (A instanceof DenseMatrix) {
			
			res = A.plus(this);
			
		} else if (A instanceof SparseMatrix) {
			
			/*
			 * When this and A are all sparse matrices,
			 * the result is also a sparse matrix.
			 */
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			
			int k1 = 0;
			int k2 = 0;
			int r1 = -1;
			int r2 = -1;
			int nzmax = 0;
			int i = -1;
			double v = 0;
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

			for (int j = 0; j < N; j++) {
				k1 = this.jc[j];
				k2 = jc[j];
				
				// Both this and A's j-th columns are empty.
				if (k1 == this.jc[j + 1] && k2 == jc[j + 1])
					continue;
				
				while (k1 < this.jc[j + 1] || k2 < jc[j + 1]) {
					
					if (k2 == jc[j + 1]) { // A's j-th column has been processed.
						i = this.ir[k1];
						v = this.pr[k1];
						k1++;
					} else if (k1 == this.jc[j + 1]) { // this j-th column has been processed.
						i = ir[k2];
						v = pr[k2];
						k2++;
					} else { // Both this and A's j-th columns have not been fully processed.
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
						map.put(Pair.of(j, i), v);
						nzmax++;
					}
					
				}
				
			}
			
			int numRows = M;
			int numColumns = N;
			ir = new int[nzmax];
			jc = new int[numColumns + 1];
			pr = new double[nzmax];
			
			int rIdx = -1;
			int cIdx = -1;
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;
			
			res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
		}
		
		return res;
		
	}

	@Override
	/**
	 * This - A.
	 */
	public Matrix minus(Matrix A) {

		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		Matrix res = null;
		
		if (A instanceof DenseMatrix) {
			
			res = A.copy();
			double[][] resData = ((DenseMatrix) res).getData();
			int r = -1;
			int k = 0;
			for (int j = 0; j < N; j++) {
				for (int i = 0; i < M; i++) {
					resData[i][j] = -resData[i][j];
				}
				for (k = jc[j]; k < jc[j + 1]; k++) {
					r = ir[k];
					// A[r][j] = pr[k]
					resData[r][j] += pr[k];
				}
			}
			
		} else if (A instanceof SparseMatrix) {
			
			/*
			 * When this and A are all sparse matrices,
			 * the result is also a sparse matrix.
			 */
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			
			int k1 = 0;
			int k2 = 0;
			int r1 = -1;
			int r2 = -1;
			int nzmax = 0;
			int i = -1;
			double v = 0;
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

			for (int j = 0; j < N; j++) {
				k1 = this.jc[j];
				k2 = jc[j];
				
				// Both this and A's j-th columns are empty.
				if (k1 == this.jc[j + 1] && k2 == jc[j + 1])
					continue;
				
				while (k1 < this.jc[j + 1] || k2 < jc[j + 1]) {
					
					if (k2 == jc[j + 1]) { // A's j-th column has been processed.
						i = this.ir[k1];
						v = this.pr[k1];
						k1++;
					} else if (k1 == this.jc[j + 1]) { // this j-th column has been processed.
						i = ir[k2];
						v = -pr[k2];
						k2++;
					} else { // Both this and A's j-th columns have not been fully processed.
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
						map.put(Pair.of(j, i), v);
						nzmax++;
					}
					
				}
				
			}
			
			int numRows = M;
			int numColumns = N;
			ir = new int[nzmax];
			jc = new int[numColumns + 1];
			pr = new double[nzmax];
			
			int rIdx = -1;
			int cIdx = -1;
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;
			
			res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
		}
		
		return res;
		
	}

	@Override
	public Matrix times(Matrix A) {

		if (A.getRowDimension() != M || A.getColumnDimension() != N) {
			System.err.println("Dimension doesn't match.");
			return null;
		}
		
		Matrix res = null;
		
		if (A instanceof DenseMatrix) {
			
			res = A.times(this);
			
		} else if (A instanceof SparseMatrix) {
			
			/*
			 * When this and A are all sparse matrices,
			 * the result is also a sparse matrix.
			 */
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) A).getIr();
			jc = ((SparseMatrix) A).getJc();
			pr = ((SparseMatrix) A).getPr();
			
			int k1 = 0;
			int k2 = 0;
			int r1 = -1;
			int r2 = -1;
			int nzmax = 0;
			int i = -1;
			double v = 0;
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

			for (int j = 0; j < N; j++) {
				k1 = this.jc[j];
				k2 = jc[j];
				
				// If the j-th column of A or this is empty, we don't need to compute.
				if (k1 == this.jc[j + 1] || k2 == jc[j + 1])
					continue;
				
				while (k1 < this.jc[j + 1] && k2 < jc[j + 1]) {
					
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
							map.put(Pair.of(j, i), v);
							nzmax++;
						}
					} else {
						k2++;
					}
					
				}
				
			}
			
			int numRows = M;
			int numColumns = N;
			ir = new int[nzmax];
			jc = new int[numColumns + 1];
			pr = new double[nzmax];
			
			int rIdx = -1;
			int cIdx = -1;
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;
			
			res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
		}
		
		return res;
		
	}

	@Override
	public Matrix times(double v) {
		if (v == 0) {
			return new SparseMatrix(M, N);
		}
		SparseMatrix res = (SparseMatrix) this.copy();
		for (int k = 0; k < nnz; k++) {
			res.pr[k] = v * pr[k];
		}
		return res;
	}

	@Override
	public Matrix copy() {
		SparseMatrix res = new SparseMatrix();
		res.ir = ir.clone();
		res.jc = jc.clone();
		res.pr = pr.clone();
		res.ic = ic.clone();
		res.jr = jr.clone();
		res.valCSRIndices = valCSRIndices.clone();
		res.M = M;
		res.N = N;
		res.nzmax = nzmax;
		res.nnz = nnz;
		return res;
	}
	
	@Override
	public Matrix clone() {
		return this.copy();
	}

	@Override
	public Matrix plus(double v) {
		Matrix res = new DenseMatrix(M, N, v);
		double[][] data = ((DenseMatrix) res).getData();
		for (int j = 0; j < N; j++) {
			for (int k = jc[j]; k < jc[j + 1]; k++) {
				data[ir[k]][j] += pr[k];
			}
		}
		return res;
	}

	@Override
	public Matrix minus(double v) {
		return this.plus(-v);
	}

	@Override
	public Vector operate(Vector b) {
		
		if (N != b.getDim()) {
			System.err.println("Dimension does not match.");
			System.exit(1);
		}
		Vector res = null;
		
		if (b instanceof DenseVector) {
			double[] V = new double[M];
			double[] pr = ((DenseVector) b).getPr();
			double s = 0;
			int c = 0;
			for (int r = 0; r < M; r++) {
				s = 0;
				for (int k = this.jr[r]; k < this.jr[r + 1]; k++) {
					c = ic[k];
					s += this.pr[valCSRIndices[k]] * pr[c];
				}
				V[r] = s;
			}
			res = new DenseVector(V);
		} else if (b instanceof SparseVector) {
			int[] ir = ((SparseVector) b).getIr();
			double[] pr = ((SparseVector) b).getPr();
			int nnz = ((SparseVector) b).getNNZ();
			double s = 0;
			int kl = 0;
			int kr = 0;
			int cl = 0;
			int rr = 0;
			TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
			for (int i = 0; i < M; i++) {
				kl = jr[i];
				kr = 0;
				s = 0;
				while (true) {
					if (kl >= jr[i + 1] || kr >= nnz) {
						break;
					}
					cl = ic[kl];
					rr = ir[kr];
					if (cl < rr) {
						kl++;
					} else if (cl > rr) {
						kr++;
					} else {
						s += this.pr[valCSRIndices[kl]] * pr[kr];
						kl++;
						kr++;
					}
				}

				if (s != 0) {
					map.put(i, s);
				}	
			}
			nnz = map.size();
			ir = new int[nnz];
			pr = new double[nnz];
			int ind = 0;
			for (Entry<Integer, Double> entry : map.entrySet()) {
				ir[ind] = entry.getKey();
				pr[ind] = entry.getValue();
				ind++;
			}
			res = new SparseVector(ir, pr, nnz, M);
		
		}
		
		return res;
		
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(100);
		if (this.getNNZ() == 0) {
			sb.append("Empty sparse matrix." + System.lineSeparator());
			return sb.toString();
		} 
		int p = 4;
		SparseMatrix S = this;
		int[] ic = S.getIc();
		int[] jr = S.getJr();
		double[] pr = S.getPr();
		int[] valCSRIndices = S.getValCSRIndices();
		int M = S.getRowDimension();
		String valueString = "";
		for (int r = 0; r < M; r++) {
			sb.append("  ");
			int currentColumn = 0;
			int lastColumn = -1;
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				currentColumn = ic[k];
				while (lastColumn < currentColumn - 1) {
					sb.append(sprintf(String.format("%%%ds", 8 + p - 4), " "));
					sb.append("  ");
					lastColumn++;
				}
				lastColumn = currentColumn;
				double v = pr[valCSRIndices[k]];
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

	@Override
	public void clear() {
		this.nzmax = 0;
		this.nnz = 0;
		this.jc = new int[N + 1];
		for (int j = 0; j < N + 1; j++) {
			jc[j] = 0;
		}
		this.jr = new int[M + 1];
		for (int i = 0; i < M + 1; i++) {
			jr[i] = 0;
		}
		ir = new int[0];
		pr = new double[0];
		ic = new int[0];
		valCSRIndices = new int[0];
	}
	
	/**
	 * Clean entries so that zero entries are removed.
	 */
	public void clean() {
		TreeSet<Pair<Integer, Integer>> set = new TreeSet<Pair<Integer, Integer>>();
		for (int k = 0; k < nnz; k++) {
			if (pr[k] == 0) {
				set.add(Pair.of(ir[k], ic[k]));
			}
		}
		for(Pair<Integer, Integer> pair : set) {
			setEntry(pair.first, pair.second, 0);
		}
	}
	
	/**
	 * Append an empty row at the end of this sparse matrix.
	 */
	public void appendAnEmptyRow() {
		int[] jr = new int[M + 2];
		System.arraycopy(this.jr, 0, jr, 0, M + 1);
		jr[M + 1] = this.jr[M];
		M++;
		this.jr = jr;
	}
	
	/**
	 * Append an empty column at the end of this sparse matrix.
	 */
	public void appendAnEmptyColumn() {
		int[] jc = new int[N + 2];
		System.arraycopy(this.jc, 0, jc, 0, N + 1);
		jc[N + 1] = this.jc[N];
		N++;
		this.jc = jc;
	}


	@Override
	public Matrix getSubMatrix(int startRow, int endRow, int startColumn,
			int endColumn) {
		int nnz = 0;
		int numRows = endRow - startRow + 1;
		int numColumns = endColumn - startColumn + 1;
		// Calculate nnz
		int rowIdx = -1;
		for (int j = startColumn; j <= endColumn; j++) {
			for (int k = this.jc[j]; k < this.jc[j + 1]; k++) {
				rowIdx = this.ir[k];
				if (rowIdx < startRow)
					continue;
				else if (rowIdx > endRow)
					break;
				else
					nnz++;
			}
		}
		
		int nzmax = nnz;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = startColumn;
		for (int j = startColumn; j <= endColumn; j++) {
			for (int t = this.jc[j]; t < this.jc[j + 1]; t++) {
				rowIdx = this.ir[t];
				if (rowIdx < startRow)
					continue;
				else if (rowIdx > endRow)
					break;
				else {
					rIdx = rowIdx - startRow;
					cIdx = j;
					pr[k] = this.pr[t];
					ir[k] = rIdx;
					while (currentColumn < cIdx) {
						jc[currentColumn + 1 - startColumn] = k;
						currentColumn++;
					}
					k++;
				}
			}
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1 - startColumn] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}


	@Override
	public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns) {
		int nRow = selectedRows.length;
		int nCol = selectedColumns.length;
		double v = 0;
		int r, c;
		SparseMatrix res = new SparseMatrix(nRow, nCol);
		for (int j = 0; j < nCol; j++) {
			c = selectedColumns[j];
			for (int i = 0; i < nRow; i++) {
				r = selectedRows[i];
				v = this.getEntry(r, c);
				if (v != 0)
					res.setEntry(i, j, v);
			}
		}
		return res;
	}


	@Override
	public Matrix getColumnMatrix(int c) {
		int nnz = this.jc[c + 1] - this.jc[c];
		if (nnz == 0)
			return new SparseMatrix(M, 1);
		int[] ir = new int[nnz];
		int[] jc = new int[] {0, nnz};
		double[] pr = new double[nnz];
		for (int k = this.jc[c], i = 0; k < this.jc[c + 1]; k++, i++) {
			ir[i] = this.ir[k];
			pr[i] = this.pr[k];
		}
		return createSparseMatrixByCSCArrays(ir, jc, pr, M, 1, nnz);
	}


	@Override
	public Vector getColumnVector(int c) {
		int dim = M;
		int nnz = jc[c + 1] - jc[c];
		if (nnz == 0)
			return new SparseVector(dim);
		int[] ir = new int[nnz];
		double[] pr = new double[nnz];
		for (int k = jc[c], i = 0; k < jc[c + 1]; k++, i++) {
			ir[i] = this.ir[k];
			pr[i] = this.pr[k];
		}
		return new SparseVector(ir, pr, nnz, dim);
	}


	@Override
	public Matrix getRowMatrix(int r) {
		int nnz = this.jr[r + 1] - this.jr[r];
		if (nnz == 0)
			return new SparseMatrix(1, N);
		int[] ic = new int[nnz];
		int[] jr = new int[] {0, nnz};
		double[] pr = new double[nnz];
		for (int k = this.jr[r], j = 0; k < this.jr[r + 1]; k++, j++) {
			ic[j] = this.ic[k];
			pr[j] = this.pr[valCSRIndices[k]];
		}
		return createSparseMatrixByCSRArrays(ic, jr, pr, 1, N, nnz);
	}


	@Override
	public Vector getRowVector(int r) {
		int dim = N;
		int nnz = jr[r + 1] - jr[r];
		if (nnz == 0)
			return new SparseVector(dim);
		int[] ir = new int[nnz];
		double[] pr = new double[nnz];
		for (int k = jr[r], j = 0; k < jr[r + 1]; k++, j++) {
			ir[j] = this.ic[k];
			pr[j] = this.pr[valCSRIndices[k]];
		}
		return new SparseVector(ir, pr, nnz, dim);
	}


	@Override
	public Matrix getRows(int startRow, int endRow) {
		int nnz = 0;
		int numRows = endRow - startRow + 1;
		int numColumns = N;
		// Calculate nnz
		int rowIdx = -1;
		for (int j = 0; j < numColumns; j++) {
			for (int k = this.jc[j]; k < this.jc[j + 1]; k++) {
				rowIdx = this.ir[k];
				if (rowIdx < startRow)
					continue;
				else if (rowIdx > endRow)
					break;
				else
					nnz++;
			}
		}
		
		int nzmax = nnz;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (int j = 0; j < numColumns; j++) {
			for (int t = this.jc[j]; t < this.jc[j + 1]; t++) {
				rowIdx = this.ir[t];
				if (rowIdx < startRow)
					continue;
				else if (rowIdx > endRow)
					break;
				else {
					rIdx = rowIdx - startRow;
					cIdx = j;
					pr[k] = this.pr[t];
					ir[k] = rIdx;
					while (currentColumn < cIdx) {
						jc[currentColumn + 1] = k;
						currentColumn++;
					}
					k++;
				}
			}
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}


	@Override
	public Matrix getRows(int... selectedRows) {
		int nRow = selectedRows.length;
		int nCol = N;
		double v = 0;
		int r, c;
		SparseMatrix res = new SparseMatrix(nRow, nCol);
		for (int j = 0; j < nCol; j++) {
			c = j;
			for (int i = 0; i < nRow; i++) {
				r = selectedRows[i];
				v = this.getEntry(r, c);
				if (v != 0)
					res.setEntry(i, j, v);
			}
		}
		return res;
	}


	@Override
	public Vector[] getRowVectors(int startRow, int endRow) {
		int numRows = endRow - startRow + 1;
		Vector[] res = new Vector[numRows];
		for (int r = startRow, i = 0; r <= endRow; r++, i++) {
			res[i] = getRowVector(r);
		}
		return res;
	}


	@Override
	public Vector[] getRowVectors(int... selectedRows) {
		int numRows = selectedRows.length;
		Vector[] res = new Vector[numRows];
		for (int i = 0; i < numRows; i++) {
			res[i] = getRowVector(selectedRows[i]);
		}
		return res;
	}


	@Override
	public Matrix getColumns(int startColumn, int endColumn) {
		int nnz = 0;
		int numRows = M;
		int numColumns = endColumn - startColumn + 1;
		// Calculate nnz
		int rowIdx = -1;
		for (int j = startColumn; j <= endColumn; j++) {
			nnz += this.jc[j + 1] - this.jc[j];
		}
		
		int nzmax = nnz;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = startColumn;
		for (int j = startColumn; j <= endColumn; j++) {
			for (int t = this.jc[j]; t < this.jc[j + 1]; t++) {
				rowIdx = this.ir[t];
				rIdx = rowIdx;
				cIdx = j;
				pr[k] = this.pr[t];
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1 - startColumn] = k;
					currentColumn++;
				}
				k++;
			}
		}
		while (currentColumn <= endColumn) {
			jc[currentColumn + 1 - startColumn] = k;
			currentColumn++;
		}
		// jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}


	@Override
	public Matrix getColumns(int... selectedColumns) {
		int nnz = 0;
		int numRows = M;
		int numColumns = selectedColumns.length;
		// Calculate nnz
		int rowIdx = -1;
		int j = -1;
		for (int c = 0; c < numColumns; c++) {
			j = selectedColumns[c];
			nnz += this.jc[j + 1] - this.jc[j];
		}
		
		int nzmax = nnz;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		// int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		for (int c = 0; c < numColumns; c++) {
			j = selectedColumns[c];
			jc[c + 1] = jc[c] + this.jc[j + 1] - this.jc[j];
			for (int t = this.jc[j]; t < this.jc[j + 1]; t++) {
				rowIdx = this.ir[t];
				rIdx = rowIdx;
				// cIdx = c;
				pr[k] = this.pr[t];
				ir[k] = rIdx;	
				k++;
			}
		}
		// jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}


	@Override
	public Vector[] getColumnVectors(int startColumn, int endColumn) {
		int numColumns = endColumn - startColumn + 1;
		Vector[] res = new Vector[numColumns];
		for (int c = startColumn, i = 0; c <= endColumn; c++, i++) {
			res[i] = getColumnVector(c);
		}
		return res;
	}


	@Override
	public Vector[] getColumnVectors(int... selectedColumns) {
		int numColumns = selectedColumns.length;
		Vector[] res = new Vector[numColumns];
		for (int j = 0; j < numColumns; j++) {
			res[j] = getColumnVector(selectedColumns[j]);
		}
		return res;
	}


	@Override
	public void setRowMatrix(int r, Matrix A) {
		if (A.getRowDimension() != 1) {
			err("Input matrix should be a row matrix.");
			exit(1);
		}
		for (int j = 0; j < N; j++) {
			setEntry(r, j, A.getEntry(0, j));
		}
	}


	@Override
	public void setRowVector(int r, Vector V) {
		for (int j = 0; j < N; j++) {
			setEntry(r, j, V.get(j));
		}
	}


	@Override
	public void setColumnMatrix(int c, Matrix A) {
		if (A.getColumnDimension() != 1) {
			err("Input matrix should be a column matrix.");
			exit(1);
		}
		for (int i = 0; i < M; i++) {
			setEntry(i, c, A.getEntry(i, 0));
		}
	}


	@Override
	public void setColumnVector(int c, Vector V) {
		for (int i = 0; i < M; i++) {
			setEntry(i, c, V.get(i));
		}
	}

}
