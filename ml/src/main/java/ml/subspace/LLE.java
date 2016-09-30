package ml.subspace;

import static la.utils.ArrayOperator.colon;
import static la.utils.InPlaceOperator.timesAssign;
import static la.utils.Matlab.diag;
import static la.utils.Matlab.eigs;
import static la.utils.Matlab.eye;
import static la.utils.Matlab.find;
import static la.utils.Matlab.getColumns;
import static la.utils.Matlab.getRows;
import static la.utils.Matlab.gt;
import static la.utils.Matlab.mrdivide;
import static la.utils.Matlab.ones;
import static la.utils.Matlab.size;
import static la.utils.Matlab.sparseRowVectors2SparseMatrix;
import static la.utils.Matlab.sumAll;
import static la.utils.Matlab.times;
import static la.utils.Printer.disp;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.manifold.Manifold;

/***
 * Locally Linear Embedding (LLE).
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class LLE extends DimensionalityReduction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = {
				{0, 2, 3, 4}, 
				{2, 0, 4, 5}, 
				{3, 4.1, 5, 6}, 
				{2, 7, 1, 6}
				};
		Matrix X = new DenseMatrix(data).transpose();
		/*int n = 20;
		int p = 10;
		X = rand(n, p);*/

		int K = 3;
		int r = 3;
		Matrix R = LLE.run(X, K, r);
		disp("Original Data:");
		disp(X);
		disp("Reduced Data:");
		disp(R);
		
		/*double[] x = R.getRow(0);
		double[] y = R.getRow(1);
		double[] z = R.getRow(2);
		
		// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("LLE", Color.RED, x, y);
		plot.addLegend("North");
		
		plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("A 2D Plot Panel");
		frame.setContentPane(plot);
		frame.setBounds(100, 100, 500, 500);
		frame.setVisible(true);
		
		Plot3DPanel plot3D = new Plot3DPanel();
		plot3D.addScatterPlot("LLE", Color.RED, x, y, z);
		plot3D.addLegend("North");
		
		plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame3D = new JFrame("A 3D Plot Panel");
		frame3D.setContentPane(plot3D);
		frame3D.setBounds(100, 100, 500, 500);
		frame3D.setVisible(true);*/
		
	}
	
	/**
	 * Number of nearest neighbors to construct the 
	 * neighborhood graph.
	 */        
	int K;
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public LLE(int r) {
		super(r);
		
	}
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @param K number of nearest neighbors to construct
	 *          the neighborhood graph
	 *          
	 */
	public LLE(int r, int K) {
		super(r);
		this.K = K;
	}

	@Override
	public void run() {
		this.R = run(X, K, r);
	}

	/**
	 * LLE (Locally Linear Embedding).
	 * 
	 * @param X an n x d data matrix with each row being a feature vector
	 * 
	 * @param K number of nearest neighbors
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return an n x r matrix which is the r dimensional 
	 *         representation of the given n examples
	 * 
	 */
	public static Matrix run(Matrix X, int K, int r) {
		String type = "nn";
		double param = K;
		Matrix A = Manifold.adjacencyDirected(X, type, param, "euclidean");
		int N = size(X, 1);
		Matrix X_i = null;
		Matrix C_i = null;
		Matrix C = null;
		Matrix w = null;
		Matrix W = gt(A, 0);
		Matrix M = null;
		Matrix Ones = ones(K, 1);
		Matrix OnesT = ones(1, K);
		Matrix I = eye(N);
	    int[] neighborIndices = null;
	    Vector[] Ws = new Vector[N];
		for (int i = 0; i < N; i++) {
			neighborIndices = find(A.getRowVector(i));
			X_i = X.getRows(neighborIndices);
			C_i = X_i.minus(Ones.mtimes(getRows(X, i)));
			// disp(C_i);
		    C = C_i.mtimes(C_i.transpose());
		    C = C.plus(diag(diag(C)));
		    w = mrdivide(OnesT, C);
		    timesAssign(w, 1 / sumAll(w));
		    // disp(w);
		    // w = rdivide(w, sumAll(w));
		    Ws[i] = new SparseVector(neighborIndices, ((DenseMatrix) w).getData()[0], neighborIndices.length, N);
		}
		W = sparseRowVectors2SparseMatrix(Ws);
		// disp(W);
		M = I.minus(W);
		M = M.transpose().mtimes(M);
		// disp(M);
		Matrix U = eigs(M, r + 1, "sm")[0];
		/*disp(U);
		disp(eigs(M, r + 1, "sm")[1]);*/
		return times(Math.sqrt(N), getColumns(U, colon(1, r)));
		
	}

}
