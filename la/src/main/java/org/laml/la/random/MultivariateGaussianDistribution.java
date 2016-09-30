package org.laml.la.random;

import static org.laml.la.utils.Matlab.diag;
import static org.laml.la.utils.Matlab.eigs;
import static org.laml.la.utils.Matlab.eye;
import static org.laml.la.utils.Matlab.mtimes;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rand;
import static org.laml.la.utils.Matlab.repmat;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Matlab.zeros;
import static org.laml.la.utils.Printer.disp;

import java.util.Random;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;

/***
 * A Java implementation for the multivariate Gaussian 
 * distribution given mean and covariance.
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class MultivariateGaussianDistribution {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int n = 10;
		int d = 2;
		Matrix t = rand(d);
		Matrix SIGMA = plus(t.mtimes(t.transpose()), times(diag(rand(d, 1)), eye(d)));
		double theta = rand(1).getEntry(0, 0) * Math.PI;
		Matrix P = new DenseMatrix(new double[][]{{Math.cos(theta), -Math.sin(theta)}, {Math.sin(theta), Math.cos(theta)}});
		// SIGMA = new DenseMatrix(new double[][]{{10, 0}, {0, 0}});
		SIGMA = P.mtimes(SIGMA).mtimes(P.transpose());
		Matrix MU = times(3, rand(1, d));
		
		/*saveMatrix(MU, "MU");
		saveMatrix(SIGMA, "SIGMA");*/
		/*MU = loadMatrix("MU");
		SIGMA = loadMatrix("SIGMA");*/
		
		Matrix X = MultivariateGaussianDistribution.mvnrnd(MU, SIGMA, n);
		disp(X);
		
		/*double[] x = X.getColumn(0);
		double[] y = X.getColumn(1);

		// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("Multivariate Gaussian Distribution", Color.RED, x, y);
		plot.addLegend("North");
		
		plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("A plot panel");
		frame.setContentPane(plot);
		frame.setBounds(100, 100, 500, 500);
		frame.setVisible(true);*/

	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * X ~ N(u, Lambda) => Y = B * X + v ~ N(B * u + v, B * Lambda * B')
	 * Therefore, if X ~ N(0, Lambda), 
	 * then Y = B * X + MU ~ N(MU, B * Lambda * B').
	 * We only need to do the eigen decomposition: SIGMA = B * Lambda * B'.
	 * 
	 * @param MU 1 x d mean vector
	 * 
	 * @param SIGMA covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(Matrix MU, Matrix SIGMA, int cases) {
		
		int d = MU.getColumnDimension();
		
		if (MU.getRowDimension() != 1) {
			System.err.printf("MU is expected to be 1 x %d matrix!\n", d);
		}
		
		if (norm(SIGMA.transpose().minus(SIGMA)) > 1e-10)
			System.err.printf("SIGMA should be a %d x %d real symmetric matrix!\n", d);	
		
		Matrix[] eigenDecompostion = eigs(SIGMA, d, "lm");
		
		Matrix B = eigenDecompostion[0];
		Matrix Lambda = eigenDecompostion[1];
		
		/*disp(B);
		disp(Lambda);*/
		
		Matrix X = new DenseMatrix(d, cases);
		Random generator = new Random();
		double sigma = 0;
		for (int i = 0; i < d; i++) {
			sigma = Lambda.getEntry(i, i);
			if (sigma == 0) {
				X.setRowMatrix(i, zeros(1, cases));
				continue;
			}
			if (sigma < 0) {
				System.err.printf("Covariance matrix should be positive semi-definite!\n");
				System.exit(1);
			}
			for (int n = 0; n < cases; n++) {
				X.setEntry(i, n, generator.nextGaussian() * Math.pow(sigma, 0.5));
			}
		}
		
		Matrix Y = plus(mtimes(B, X), repmat(MU.transpose(), 1, cases)).transpose();
		
		return Y;
		
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * X ~ N(u, Lambda) => Y = B * X + v ~ N(B * u + v, B * Lambda * B')
	 * Therefore, if X ~ N(0, Lambda), 
	 * then Y = B * X + MU ~ N(MU, B * Lambda * B').
	 * We only need to do the eigen decomposition: SIGMA = B * Lambda * B'.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 2D {@code double} array holding the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(double[] MU, double[][] SIGMA, int cases) {
		return mvnrnd(new DenseMatrix(MU, 2), new DenseMatrix(SIGMA), cases);
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and a diagonal covariance SIGMA.
	 * 
	 * X ~ N(u, Lambda) => Y = B * X + v ~ N(B * u + v, B * Lambda * B')
	 * Therefore, if X ~ N(0, Lambda), 
	 * then Y = B * X + MU ~ N(MU, B * Lambda * B').
	 * We only need to do the eigen decomposition: SIGMA = B * Lambda * B'.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 1D {@code double} array holding the diagonal elements
	 *        of the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(double[] MU, double[] SIGMA, int cases) {
		return mvnrnd(new DenseMatrix(MU, 2), diag(SIGMA), cases);
	}

}
