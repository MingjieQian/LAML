package la.vector;

import static ml.utils.Printer.disp;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Matlab.full;
import static ml.utils.Matlab.minus;
import static ml.utils.Matlab.plus;
import static ml.utils.Matlab.sparse;
import static ml.utils.Matlab.times;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;

public class Test {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Vector V1 = new DenseVector(new double[] {2, 0, 4});
		Vector V2 = new DenseVector(new double[] {1, 2, 0});
		
		fprintf("V1 .* V2:%n");
		disp(times(V1, V2));
		
		fprintf("V1 .* sparse(V2):%n");
		disp(times(V1, sparse(V2)));
		
		fprintf("sparse(V1) .* sparse(V2):%n");
		disp(times(sparse(V1), sparse(V2)));
		
		fprintf("V1 + V2:%n");
		disp(plus(V1, V2));
		
		fprintf("V1 + sparse(V2):%n");
		disp(plus(V1, sparse(V2)));
		
		fprintf("sparse(V1) + sparse(V2):%n");
		disp(plus(sparse(V1), sparse(V2)));
		
		fprintf("V1 - V2:%n");
		disp(minus(V1, V2));
		
		fprintf("V1 - sparse(V2):%n");
		disp(minus(V1, sparse(V2)));
		
		fprintf("sparse(V1) - sparse(V2):%n");
		disp(minus(sparse(V1), sparse(V2)));
		
		int dim = 4;
		Vector V = new SparseVector(dim);
		
		for (int i = 0; i < dim; i++) {
			fprintf("V(%d):	%.2f%n", i + 1, V.get(i));
		}
		
		V.set(3, 4.5);
		fprintf("V(%d):	%.2f%n", 3 + 1, V.get(3));
		V.set(1, 2.3);
		fprintf("V(%d):	%.2f%n", 1 + 1, V.get(1));
		V.set(1, 3.2);
		fprintf("V(%d):	%.2f%n", 1 + 1, V.get(1));
		V.set(3, 2.5);
		fprintf("V(%d):	%.2f%n", 3 + 1, V.get(3));
		
		fprintf("V:%n");
		disp(V);
		
		// disp(full(V));
		
		for (int i = 0; i < dim; i++) {
			fprintf("V(%d):	%.2f%n", i + 1, V.get(i));
		}
		
		Matrix A = null;
		int[] rIndices = new int[] {0, 1, 3, 1, 2, 2, 3, 2, 3};
		int[] cIndices = new int[] {0, 0, 0, 1, 1, 2, 2, 3, 3};
		double[] values = new double[] {10, 3.2, 3, 9, 7, 8, 8, 7, 7};
		int numRows = 4;
		int numColumns = 4;
		int nzmax = rIndices.length;

		A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		
		fprintf("A:%n");
		printMatrix(A);
		
		fprintf("AV:%n");
		disp(A.operate(V));
		
		fprintf("V'A':%n");
		disp(V.operate(A.transpose()));
		
		disp(A.operate(full(V)));
		
		disp(full(A).operate(V));
		
		disp(full(A).operate(full(V)));
		
		SparseVector V3 = new SparseVector(1411);
		V3.set(67, 1);
		V3.set(1291, 0.7514);
		
		int k = 0;
		double xk = V3.get(k);
		System.out.println(xk);
		
	}

}
