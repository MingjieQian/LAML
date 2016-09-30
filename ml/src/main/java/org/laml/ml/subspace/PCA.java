package ml.subspace;

import static la.utils.ArrayOperator.divideAssign;
import static la.utils.Matlab.eigs;
import static la.utils.Matlab.size;
import static la.utils.Matlab.sum;
import static la.utils.Printer.disp;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;

/***
 * Principal Component Analysis (PCA).
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class PCA extends DimensionalityReduction {

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
		Matrix X = new DenseMatrix(data).transpose();
		/*int n = 20;
		int p = 10;
		X = rand(n, p);*/

		int r = 3;
		Matrix R = PCA.run(X, r);
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
		plot.addScatterPlot("PCA", Color.RED, x, y);
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
		plot3D.addScatterPlot("PCA", Color.RED, x, y, z);
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
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public PCA(int r) {
		super(r);
	}

	@Override
	public void run() {
		this.R = PCA.run(X, r);
	}
	
	
	/**
	 * PCA.
	 * 
	 * @param X an n x d data matrix with each row being a feature vector
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return an n x r matrix which is the r dimensional 
	 *         representation of the given n examples
	 *         
	 */
	public static Matrix run(Matrix X, int r) {
		
		int n = size(X, 1);
		double[] S = sum(X).getPr();
		divideAssign(S, n);
		X = X.copy();
		int d = X.getColumnDimension();
		double s = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				s = S[j];
				if (s != 0)
					X.setEntry(i, j, X.getEntry(i, j) - s);
			}
		}
		// X = X.subtract(repmat(mean(X, 2), 1, N));
		Matrix XT = X.transpose();
		Matrix Psi = XT.mtimes(X);
		/*disp(Psi);
		disp(eigs(Psi, r, "lm")[0]);
		disp(eigs(Psi, r, "lm")[1]);*/
		return X.mtimes(eigs(Psi, r, "lm")[0]);
		
	}

}