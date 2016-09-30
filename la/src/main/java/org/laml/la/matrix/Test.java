package la.matrix;

import static la.utils.Printer.disp;
import static la.utils.Printer.fprintf;
import static la.utils.Printer.printMatrix;
import static la.utils.Printer.printVector;
import static la.utils.Matlab.*;

import java.util.Map.Entry;
import java.util.TreeMap;

import la.decomposition.LUDecomposition;
import la.vector.DenseVector;
import la.vector.Vector;
import la.utils.Pair;
import la.utils.Matlab;

public class Test {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		TreeMap<Integer, Integer> treeMap = new TreeMap<Integer, Integer>();

		treeMap.put(5, 0);
		treeMap.put(3, 1);
		treeMap.put(6, 2);
		treeMap.put(4, 3);
		System.out.printf("Key	Value%n");
		for (Entry<Integer, Integer> entry : treeMap.entrySet()) {
			System.out.printf("%d	%d%n", entry.getKey(), entry.getValue());
		}
		System.out.println();
		
		TreeMap<Pair<Integer, Integer>, Integer> map = new TreeMap<Pair<Integer, Integer>, Integer>();
		
		map.put(Pair.of(4, 0), 0);
		map.put(Pair.of(5, 0), 1);
		map.put(Pair.of(3, 1), 2);
		map.put(Pair.of(6, 2), 3);
		map.put(Pair.of(4, 3), 4);
		System.out.printf("RowIdx	ColIdx	ValIdx%n");
		for (Entry<Pair<Integer, Integer>, Integer> entry : map.entrySet()) {
			System.out.printf("%d	%d	%d%n", entry.getKey().first, entry.getKey().second, entry.getValue());
		}
		System.out.println();
		
		/*
		 * 10	0	0	0
		 * 3	9	0	0
		 * 0	7	8	7
		 * 3	0	7	7 
		 */
		int[] rIndices = new int[] {0, 1, 3, 1, 2, 2, 3, 2, 3};
		int[] cIndices = new int[] {0, 0, 0, 1, 1, 2, 2, 3, 3};
		double[] values = new double[] {10, 3.2, 3, 9, 7, 8, 7, 7, 7};
		int numRows = 4;
		int numColumns = 4;
		int nzmax = rIndices.length;

		Matrix S = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		fprintf("S:%n");
		printMatrix(S, 4);
		
		fprintf("det(S) = %.4f\n", det(S));

		LUDecomposition LUDecomp = new LUDecomposition(S);
		disp("L:");
		disp(LUDecomp.getL());
		disp("U:");
		disp(LUDecomp.getU());
		
		Matrix invS = inv(S);
		disp("inv(S):");
		disp(invS);

		disp("invS * S:");
		disp(invS.mtimes(S));

		fprintf("S':%n");
		printMatrix(S.transpose(), 4);
		
		fprintf("S'':%n");
		printMatrix(S.transpose().transpose(), 4);
		
		fprintf("S * S':%n");
		printMatrix(S.mtimes(S.transpose()), 4);
		
		fprintf("S' * S:%n");
		printMatrix(S.transpose().mtimes(S), 4);
		
		double[][] data = {
				{3.5, 4.4, 1.3, 2.3},
				{5.3, 2.2, 0.5, 4.5},
				{0.2, 0.3, 4.1, -3.1},
				{-1.2, 0.4, 3.2, 1.6}
				};
		
		Matrix A = new DenseMatrix(data);
		fprintf("A:%n");
		printMatrix(A, 4);
		
		fprintf("A':%n");
		printMatrix(A.transpose(), 4);
		
		Matrix B = A.mtimes(S);
		fprintf("A * S:%n");
		printMatrix(B, 4);
		
		fprintf("S * A:%n");
		printMatrix(S.mtimes(A), 4);
		
		if (S instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) S).getIc();
			int[] jr = ((SparseMatrix) S).getJr();
			double[] pr = ((SparseMatrix) S).getPr();
			int[] valCSRIndices = ((SparseMatrix) S).getValCSRIndices();
			double[] pr_CSR = new double[pr.length];
			for (int k = 0; k < ((SparseMatrix) S).getNZMax(); k++){
				pr_CSR[k] = pr[valCSRIndices[k]];
			}
			Matrix S_CSR = SparseMatrix.createSparseMatrixByCSRArrays(ic, jr, pr_CSR, S.getRowDimension(), S.getColumnDimension(), ((SparseMatrix) S).getNZMax());
			fprintf("S_CSC:%n");
			printMatrix(S, 4);
			fprintf("S_CSR:%n");
			printMatrix(S_CSR, 4);
		}
		
		// int[].clone will generate a new copy
		int[] rIndices2 = rIndices.clone();
		rIndices2[0] = -1;
		
		// double[][].clone will not generate a new copy
		Matrix A_copy = A.copy();
		double[][] data2 = ((DenseMatrix) A_copy).getData().clone();
		data2[0][0] = -100;
		
		fprintf("A + S:%n");
		printMatrix(A.plus(S), 4);
		
		fprintf("S + A:%n");
		printMatrix(S.plus(A), 4);
		
		Matrix S2 = S.mtimes(S.transpose());
		
		fprintf("S:%n");
		printMatrix(S, 4);
		
		fprintf("S * S':%n");
		printMatrix(S2, 4);
		
		fprintf("S * S' + S:%n");
		printMatrix(S2.plus(S), 4);
		
		fprintf("S - S * S':%n");
		printMatrix(S.minus(S2), 4);
		
		fprintf("S .* S':%n");
		printMatrix(S.times(S.transpose()), 4);
		
		fprintf("S' .* S:%n");
		printMatrix(S.transpose().times(S), 4);
		
		Vector V = new DenseVector(new double[] {0, 1, 0, 3});
		fprintf("S * V:%n");
		printVector(S.operate(V));
		
		V = sparse(V);
		fprintf("S * V:%n");
		printVector(S.operate(V));
		
		fprintf("S:%n");
		printMatrix(S, 4);
		
		fprintf("S:%n");
		disp(S, 4);
		
		int r = -1;
		int c = -1;
		double v = 0;
		
		r = 3;
		c = 1;
		v = 1.5;
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		fprintf("S(%d, %d) <- %.2f%n", r + 1, c + 1, v);
		S.setEntry(r, c, v);
		fprintf("S:%n");
		printMatrix(S, 4);
		fprintf("S:%n");
		disp(S);
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		
		r = 0;
		c = 2;
		v = 2.5;
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		fprintf("S(%d, %d) <- %.2f%n", r + 1, c + 1, v);
		S.setEntry(r, c, v);
		fprintf("S:%n");
		printMatrix(S, 4);
		fprintf("S:%n");
		disp(S);
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		
		v = 0;
		fprintf("Delete S(%d, %d)%n", r + 1, c + 1);
		fprintf("S(%d, %d) <- %.2f%n", r + 1, c + 1, v);
		S.setEntry(r, c, v);
		fprintf("S:%n");
		printMatrix(S, 4);
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		
		r = 3;
		c = 1;
		v = 0;
		fprintf("Delete S(%d, %d)%n", r + 1, c + 1);
		fprintf("S(%d, %d) <- %.2f%n", r + 1, c + 1, v);
		S.setEntry(r, c, v);
		fprintf("S:%n");
		printMatrix(S, 4);
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		
		r = 3;
		c = 1;
		v = 1.5;
		fprintf("Add S(%d, %d) = %.2f%n", r + 1, c + 1, v);
		fprintf("S(%d, %d) <- %.2f%n", r + 1, c + 1, v);
		S.setEntry(r, c, v);
		fprintf("S:%n");
		printMatrix(S, 4);
		fprintf("S(%d, %d) = %.2f%n", r + 1, c + 1, S.getEntry(r, c));
		
		Matrix S3 = S.copy();
		
		int M = 5;
		int N = 5;
		S = new SparseMatrix(M, N);
		fprintf("S:%n");
		disp(S);

		S.setEntry(0, 3, 0.9);
		S.setEntry(0, 1, 0.8);
		S.setEntry(0, 1, 3.2);
		S.setEntry(0, 3, 0.6);
		/*S.setEntry(3, 0, 0.9);
		S.setEntry(1, 0, 0.8);
		S.setEntry(1, 0, 3.2);*/
		// S.setEntry(2, 3, 1.8);
		// S.setEntry(2, 3, 2.0);
		
		fprintf("S:%n");
		disp(S);
		
		S = S3;
		fprintf("S:%n");
		printMatrix(S);
		fprintf("S:%n");
		disp(S);
		Vector[] Vs = null;
		Vs = Matlab.sparseMatrix2SparseRowVectors(S);
		for (int i = 0; i < Vs.length; i++) {
			fprintf("Vs[%d]:%n", i);
			disp(Vs[i]);
		}
		printMatrix(Matlab.sparseRowVectors2SparseMatrix(Vs));
		disp(Matlab.sparseRowVectors2SparseMatrix(Vs));
		
		fprintf("S:%n");
		printMatrix(S);
		Vs = Matlab.sparseMatrix2SparseColumnVectors(S);
		for (int i = 0; i < Vs.length; i++) {
			fprintf("Vs[%d]:%n", i);
			disp(Vs[i]);
		}
		printMatrix(Matlab.sparseColumnVectors2SparseMatrix(Vs));
		disp(Matlab.sparseColumnVectors2SparseMatrix(Vs));
		
		fprintf("S:%n");
		printMatrix(S);
		
		((SparseMatrix) S).appendAnEmptyRow();
		fprintf("Append an empty row to S:%n");
		printMatrix(full(S));
		
		((SparseMatrix) S).appendAnEmptyColumn();
		fprintf("Append an empty column to S:%n");
		printMatrix(full(S));
		
		disp(hilb(5, 5));
		printMatrix(hilb(5, 5).times(S));
		
	}

}
