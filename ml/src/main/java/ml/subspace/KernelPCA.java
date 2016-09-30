package ml.subspace;

import static la.utils.ArrayOperator.timesAssign;
import static la.utils.Matlab.eigs;
import static la.utils.Matlab.eye;
import static la.utils.Matlab.ones;
import static la.utils.Matlab.rdivide;
import static la.utils.Matlab.size;
import static la.utils.Printer.disp;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.kernel.Kernel;

/***
 * Kernel PCA
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class KernelPCA extends DimensionalityReduction {

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
		Matrix R = KernelPCA.run(X, r);
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
		plot.addScatterPlot("KernelPCA", Color.RED, x, y);
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
		plot3D.addScatterPlot("KernelPCA", Color.RED, x, y, z);
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
	public KernelPCA(int r) {
		super(r);

	}

	@Override
	public void run() {
		this.R = run(X, r);
	}
	
	/**
	 * Kernel PCA.
	 * 
	 * @param X an n x d data matrix
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return an n x r matrix which is the r dimensional 
	 *         representation of the given n examples
	 *         
	 */
	public static Matrix run(Matrix X, int r) {
		
		int N = size(X, 1);
		
		Matrix H = eye(N).minus(rdivide(ones(N, N), N));
		double sigma = 1.0;
		Matrix K = Kernel.calcKernel("rbf", sigma, X);
		Matrix Psi = H.mtimes(K).mtimes(H);
		// IO.saveMatrix(Psi, "Psi");
		// disp(Psi);
		Matrix[] UD = eigs(Psi, r, "lm");
		Matrix U = UD[0];
		Matrix D = UD[1];
		/*disp(U);
		disp(D);
		disp(U.transpose().mtimes(U));
		disp(Psi.mtimes(U));
		disp(U.mtimes(D));
		disp("UDU':");
		disp(U.mtimes(D).mtimes(U.transpose()));*/
		/*D = diag(rdivide(1, pow(diag(D), 0.5)));
		U = mtimes(U, D);*/
		double[] s = new double[r];
		for (int j = 0; j < r; j++) {
			s[j] = 1 / Math.sqrt(D.getEntry(j, j));
		}
		double[][] eigData = ((DenseMatrix) U).getData();
		for (int i = 0; i < N; i++) {
			timesAssign(eigData[i], s);
		}
		
		return K.mtimes(U);
		
	}

}
