package ml.subspace;

import static la.utils.Matlab.eq;
import static la.utils.Matlab.logicalIndexingAssignment;
import static la.utils.Matlab.size;
import static la.utils.Matlab.speye;
import static la.utils.Printer.disp;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.manifold.Manifold;

/***
 * Isomap.
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class Isomap extends DimensionalityReduction {

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
		/*double[][] data = { 
				{0, 2, 3, 4, 5}, 
				{2, 0, 4, 5, 6}, 
				{3, 4.1, 5, 6, 7}, 
				{2, 7, 1, 6, 8}, 
				{2, 7, 1, 4, 8}, 
				{2, 7, 1, 3, 8} 
		};*/
		Matrix X = new DenseMatrix(data);
		X = X.transpose();
		/*int n = 20;
		int p = 10;
		X = rand(n, p);*/

		int K = 3;
		int r = 3;
		Matrix R = Isomap.run(X, K, r);
		disp("Original Data:");
		disp(X);
		disp("Reduced Data:");
		disp(R);
		
		/*double[] x = R.getRow(0);
		double[] y = R.getRow(1);
		double[] z = R.getRow(2);*/
		
		/*// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("Isomap", Color.RED, x, y);
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
		plot3D.addScatterPlot("Isomap", Color.RED, x, y, z);
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
	public Isomap(int r) {
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
	public Isomap(int r, int K) {
		super(r);
		this.K = K;
	}

	@Override
	public void run() {
		this.R = run(X, K, r);
	}
	
	/**
	 * Isomap (isometric feature mapping).
	 * 
	 * @param X an n x d data matrix with each row being a feature vector
	 * 
	 * @param K number of nearest neighbors
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return an n x r matrix which is the r dimensional 
	 *         representation of the given n data examples
	 * 
	 */
	public static Matrix run(Matrix X, int K, int r) {
		
		// Step 1: Construct neighborhood graph
		Matrix D = Manifold.adjacency(X, "nn", K, "euclidean");
		logicalIndexingAssignment(D, eq(D, 0), Double.POSITIVE_INFINITY);
		logicalIndexingAssignment(D, speye(size(D)), 0);
		
		// Step 2: Compute shortest paths
		int d = size(D, 1);
		for (int k = 0; k < d; k++) {
			for (int i = 0; i < d; i++) {
				for (int j = 0; j < d; j++) {
					D.setEntry(i, j, Math.min(D.getEntry(i,j), D.getEntry(i,k) + D.getEntry(k,j)));
				}
			}
		}
		
		// Construct r-dimensional embedding
		return MDS.run(D, r);
		
	}

}
