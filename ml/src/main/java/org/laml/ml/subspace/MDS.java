package org.laml.ml.subspace;

import static org.laml.la.utils.ArrayOperator.allocate2DArray;
import static org.laml.la.utils.ArrayOperator.timesAssign;
import static org.laml.la.utils.Matlab.eigs;
import static org.laml.la.utils.Matlab.eye;
import static org.laml.la.utils.Matlab.l2Distance;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.ones;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Printer.disp;
import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;

/***
 * Multi-dimensional Scaling (MDS).
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class MDS extends DimensionalityReduction {

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
		Matrix O = new DenseMatrix(data).transpose();
		/*int n = 20;
		int p = 10;
		O = rand(n, p);*/

		Matrix D = l2Distance(O, O);
		// disp(D.minus(D.transpose()));
		// fprintf("%g\n", norm(D.minus(D.transpose())));
		Matrix X = MDS.run(D, 3);
		disp("Reduced X:");
		disp(X);
		
		/*double[] x = X.getRow(0);
		double[] y = X.getRow(1);
		double[] z = X.getRow(2);
		
		// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("MultiDimensional Scaling", Color.RED, x, y);
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
		plot3D.addScatterPlot("MultiDimensional Scaling", Color.RED, x, y, z);
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
	public MDS(int r) {
		super(r);
	}

	@Override
	public void run() {
		
	}
	
	/**
	 * Dimensionality reduction by MDS.
	 * 
	 * @param D an n x n dissimilarity matrix where n is the 
	 *          sample size
	 *          
	 * @param p number of dimensions to be reduced to
	 * 
	 * @return an n x p matrix which is the p dimensional 
	 *         representation of the given n objects with dissimilarity 
	 *         matrix D
	 *             
	 */
	public static Matrix run(Matrix D, int p) {
		/*disp(norm(zeros(3, 3)));
		disp(norm(D.minus(D.transpose())));
		double norm = norm(D.minus(D.transpose()));
		boolean flag = Double.isNaN(norm);
		if (flag) {
			Matrix DDT = D.minus(D.transpose());
			norm(DDT);
		}*/
		if (norm(D.minus(D.transpose())) > 1e-12) {
			System.err.println("The dissimilarity matrix should be symmetric!");
			System.exit(1);
		}
		int n = D.getColumnDimension();
		Matrix A = times(-1d/2, times(D, D));
		Matrix H = eye(n).minus(rdivide(ones(n), n));
		Matrix B = H.mtimes(A).mtimes(H);
		B = rdivide(plus(B, B.transpose()), 2);
		// fprintf("%g\n", norm(B.minus(B.transpose())));
		Matrix[] eigRes = eigs(B, n, "lm");
		int k = 0;
		for (k = p - 1; k >= 0; k--) {
			if (eigRes[1].getEntry(k, k) > 0)
				break;
		}
		double[][] eigData = ((DenseMatrix) eigRes[0]).getData();
		double[][] resData = allocate2DArray(n, k + 1);
		double[] resRow = null;
		double[] s = new double[k + 1];
		for (int j = 0; j <= k; j++) {
			s[j] = Math.sqrt(eigRes[1].getEntry(j, j));
		}
		for (int i = 0; i < n; i++) {
			resRow = resData[i];
			System.arraycopy(eigData[i], 0, resRow, 0, k + 1);
			timesAssign(resRow, s);
		}
		return new DenseMatrix(resData);
		
	}

}
