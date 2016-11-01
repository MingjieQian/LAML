package org.laml.ml.temporal;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.ml.optimization.AcceleratedProximalGradient;
import org.laml.ml.optimization.ProxPlus;
import org.laml.ml.options.Options;
import org.laml.ml.regression.LASSO;
import org.laml.ml.regression.Regression;
import org.laml.la.utils.InPlaceOperator;
import org.laml.la.utils.Matlab;
import org.laml.la.utils.Printer;

public class TemporalLASSO extends SurvivalScoreModel {

	public static void main(String[] args) {

	}
	
	private int gType;
	
	private double theta;
	
	private double rho;
	
	private double mu;
	
	private Options options;
	
	public TemporalLASSO() {
		
	}
	
	public TemporalLASSO(Options options, int gType) {
		this.options = options;
		this.gType = gType;
	}
	
	public void initialize(double theta0, double rho0, double mu0) {
		this.theta = theta0;
		this.rho = rho0;
		this.mu = mu0;
	}
	
	public void initialize(double... params) {
		if (params.length == 3) {
			theta = params[0];
			rho = params[1];
			mu = params[2];
		} else if (params.length == 2) {
			rho = params[0];
			mu = params[1];
		} else if (params.length == 1) {
			theta = params[0];
		}
	}
	
	public void setParams(double... params) {
		if (params.length == 3) {
			theta = params[0];
			rho = params[1];
			mu = params[2];
		} else if (params.length == 2) {
			rho = params[0];
			mu = params[1];
		} else if (params.length == 1) {
			theta = params[0];
		}
	}

	@Override
	public void train() {
		int ny = Y.getColumnDimension();
		int maxIter = options.maxIter;
		if (Double.isInfinite(theta))
			theta = Matlab.sumAll(Y) / n;
		if (Double.isInfinite(rho))
			rho = 1;
		if (Double.isInfinite(mu))
			mu = 2.5;
		double[] params = null;
		Matrix Xg = X.copy();
		double lambda = this.options.lambda;
		Options options = new Options();
		options.maxIter = 50;
		options.lambda = n * lambda;
		options.verbose = this.options.verbose;
		options.calc_OV = this.options.calc_OV;
		options.epsilon = this.options.epsilon;
		Regression lasso = new LASSO(options);
		W = Matlab.zeros(p, ny);
		Matrix Theta = null;
		Matrix XW = Matlab.zeros(Matlab.size(Y));
		
		AcceleratedProximalGradient.prox = new ProxPlus();
		AcceleratedProximalGradient.type = 0;
		Matrix ThetaGrad = null;
		switch(gType) {
		case 1:
			params = new double[] {theta};
			Theta = Matlab.ones(1, 1);
			ThetaGrad = Matlab.ones(1, 1);
			break;
		case 2:
			params = new double[] {rho, mu};
			Theta = Matlab.ones(2, 1);
			ThetaGrad = Matlab.ones(2, 1);
			break;
		case 3:
			params = new double[] {theta};
			Theta = Matlab.ones(1, 1);
			ThetaGrad = Matlab.ones(1, 1);
			break;
		case 5:
			params = new double[] {rho, mu};
			Theta = Matlab.ones(2, 1);
			ThetaGrad = Matlab.ones(2, 1);
			break;
		case 6:
			params = new double[] {theta};
			Theta = Matlab.ones(1, 1);
			ThetaGrad = Matlab.ones(1, 1);
			break;
		case 7:
			params = new double[] {rho};
			Theta = Matlab.ones(1, 1);
			ThetaGrad = Matlab.ones(1, 1);
			break;
		default:
		}
		
		double fval = 0;
		double fval_pre = 0;
		int cnt = 0;
		while(true) {
			
			// Update W:
			Matrix gVal = g(gType, T, params);
			InPlaceOperator.mtimes(Xg, Matlab.diag(gVal), X);
			W = lasso.train(Xg, Y, W);
			
			// Update theta:
			boolean flags[] = null;
			double epsilon = 1e-3;
			int k = 0;
			double gval = 0;
			int APGMaxIter = 1000;
			double hval = 0;
			
			InPlaceOperator.mtimes(XW, X, W);
			InPlaceOperator.mtimes(ThetaGrad, dg(gType, T, params), XW.times(g(gType, T, params)).minus(Y).times(XW));
			InPlaceOperator.timesAssign(ThetaGrad, 2.0 / n);
			
			gval = Matlab.norm(Y.minus(Xg.mtimes(W)), "fro");
			gval = gval * gval / n;
			
			Theta.setEntry(0, 0, params[0]);
			if (gType == 2 || gType == 5)
				Theta.setEntry(1, 0, params[1]);
			
			while (true) {

				flags = AcceleratedProximalGradient.run(ThetaGrad, gval, hval, epsilon, Theta);
				// flags = NonnegativePLBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = LBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = AcceleratedGradientDescent.run(ThetaGrad, gval, epsilon, Theta);

				if (flags[0])
					break;
				params[0] = Theta.getEntry(0, 0);
				if (gType == 2 || gType == 5)
					params[1] = Theta.getEntry(1, 0);
				gVal = g(gType, T, params);
				InPlaceOperator.mtimes(Xg, Matlab.diag(gVal), X);
				gval = Matlab.norm(Y.minus(Xg.mtimes(W)), "fro");
				gval = gval * gval / n;
				
				/*
				 *  Compute the objective function value, if flags[1] is true
				 *  gradient will also be computed.
				 */
				if (flags[1]) {
					
					k = k + 1;
					// Printer.fprintf("Iter %d - gval: %.4f\n", k, gval);
					
					// Compute the gradient
					if (k > APGMaxIter)
						break;
					
					InPlaceOperator.mtimes(ThetaGrad, dg(gType, T, params), XW.times(g(gType, T, params)).minus(Y).times(XW));
					InPlaceOperator.timesAssign(ThetaGrad, 2.0 / n);
					
				}
				
			}
			params[0] = Theta.getEntry(0, 0);
			if (gType == 2 || gType == 5)
				params[1] = Theta.getEntry(1, 0);
			cnt++;
			fval = gval + lambda * Matlab.norm(W, 1);
			Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
			if ( cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps)
				break;
			fval_pre = fval;
			if (cnt > maxIter)
				break;
		}
		
		switch(gType) {
		case 1:
			theta = params[0];
			break;
		case 2:
			rho = params[0];
			mu = params[1];
			break;
		case 3:
			theta = params[0];
			break;
		case 5:
			rho = params[0];
			mu = params[1];
			break;
		case 6:
			theta = params[0];
			break;
		case 7:
			rho = params[0];
			break;
		default:
		}
		
	}
	
	static Matrix dg(int gType, Matrix T, double[] params) {
		int n = T.getRowDimension();
		double[][] resData = null;
		switch(gType) {
		case 1: {
			resData = new double[1][n];
			resData[0] = new double[n];
			double theta = params[0];
			Matrix TSq = T.times(T);
			Matrix TCu = Matlab.pow(T, 3);
			double thetaCu = Math.pow(theta, 3);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tSq = TSq.getEntry(i, 0);
				double tCu = TCu.getEntry(i, 0);
				resData[0][i] = 6 * tSq * (tSq - thetaCu) / Math.pow(thetaCu + 2 * tCu, 2);
			}
			break;
		} 
		case 2: {
			resData = new double[2][n];
			resData[0] = new double[n];
			resData[1] = new double[n];
			double rho = params[0];
			double mu = params[1];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				double emrhot = Math.exp(-rho * t);
				resData[0][i] = t * emrhot * Math.cos(mu * t);
				resData[1][i] = t * emrhot * Math.sin(mu * t);
			}
			break;
		}
		case 3: {
			resData = new double[1][n];
			resData[0] = new double[n];
			double theta = params[0];
			Matrix TSq = T.times(T);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tsq = TSq.getEntry(i, 0);
				resData[0][i] = -tsq / Math.pow(theta + tsq, 2);
			}
			break;
		}
		case 5: {
			resData = new double[2][n];
			resData[0] = new double[n];
			resData[1] = new double[n];
			double rho = params[0];
			double mu = params[1];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				double tmmu = t - mu;
				double erhotmmu = Math.exp(rho * tmmu);
				double denominator = Math.pow(1 + Math.exp(rho * (t - mu)), 2);
				resData[0][i] = -tmmu * erhotmmu / denominator;
				resData[1][i] = rho * erhotmmu / denominator;
			}
			break;
		}
		case 6: {
			resData = new double[1][n];
			resData[0] = new double[n];
			double theta = params[0];
			Matrix TSq = T.times(T);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tsq = TSq.getEntry(i, 0);
				resData[0][i] = tsq / Math.pow(theta + tsq, 2);
			}
			break;
		}
		case 7: {
			resData = new double[1][n];
			resData[0] = new double[n];
			double rho = params[0];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				resData[0][i] = -t * Math.exp(-rho * t);
			}
			break;
		}
		default:
		}
		return new DenseMatrix(resData);
	}
	
	static Matrix g(int gType, Matrix T, double[] params) {
		Matrix res = T.copy();
		switch(gType) {
		case 1: {
			double theta = params[0];
			Matrix TSq = T.times(T);
			Matrix TCu = Matlab.pow(T, 3);
			double thetaCu = Math.pow(theta, 3);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tSq = TSq.getEntry(i, 0);
				double tCu = TCu.getEntry(i, 0);
				res.setEntry(i, 0, 3 * theta * tSq / (thetaCu + 2 * tCu));
			}
			break;
		} 
		case 2: {
			double rho = params[0];
			double mu = params[1];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				res.setEntry(i, 0, 1 - Math.exp(-rho * t) * Math.cos(mu * t));
			}
			break;
		}
		case 3: {
			double theta = params[0];
			Matrix TSq = T.times(T);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tsq = TSq.getEntry(i, 0);
				res.setEntry(i, 0, tsq / (theta + tsq));
			}
			break;
		}
		case 5: {
			double rho = params[0];
			double mu = params[1];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				res.setEntry(i, 0, 1 / (1 + Math.exp(rho * (t - mu))));
			}
			break;
		}
		case 6: {
			double theta = params[0];
			Matrix TSq = T.times(T);
			for (int i = 0; i < T.getRowDimension(); i++) {
				double tsq = TSq.getEntry(i, 0);
				res.setEntry(i, 0, theta / (theta + tsq));
			}
			break;
		}
		case 7: {
			double rho = params[0];
			for (int i = 0; i < T.getRowDimension(); i++) {
				double t = T.getEntry(i, 0);
				res.setEntry(i, 0, Math.exp(-rho * t));
			}
			break;
		}
		default:
		}
		return res;
	}

	@Override
	public Matrix predict(Matrix Xt, Matrix Tt) {
		double[] params = new double[2];
		switch(gType) {
		case 1:
			params = new double[] {theta};
			break;
		case 2:
			params = new double[] {rho, mu};
			break;
		case 3:
			params = new double[] {theta};
			break;
		case 5:
			params = new double[] {rho, mu};
			break;
		case 6:
			params = new double[] {theta};
			break;
		case 7:
			params = new double[] {rho};
			break;
		default:
		}
		Matrix gVal = g(gType, Tt, params);
		Matrix PredY = Xt.mtimes(W).times(gVal);
		return PredY;
	}

	@Override
	public void loadModel(String filePath) {

		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			double[] params = (double[])ois.readObject();
			setParams(params);
			gType = (Integer)ois.readObject();
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

	}

	@Override
	public void saveModel(String filePath) {

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.writeObject(new double[] {theta, rho, mu});
			oos.writeObject(new Integer(gType));
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
