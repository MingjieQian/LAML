package ml.temporal;

import static ml.utils.InPlaceOperator.mtimes;
import static ml.utils.InPlaceOperator.operate;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Matlab.norm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.DenseVector;
import ml.optimization.AcceleratedProximalGradient;
import ml.optimization.ProxL1;
import ml.optimization.ProxPlus;
import ml.optimization.ProximalMapping;
import ml.options.Options;
import ml.utils.Matlab;
import ml.utils.Printer;

public class ReflectedLogisticRegression extends SurvivalScoreModel {

	public static void main(String[] args) {

	}
	
	private double rho;
	
	private Options options;
	
	public ReflectedLogisticRegression(double lambda) {
		options.lambda = lambda;
	}
	
	public ReflectedLogisticRegression(Options options) {
		this.options = options;
	}
	
	public void initialize(double rho0) {
		this.rho = rho0;
	}
	
	public void initialize(double... params) {
		if (params.length == 3) {
			rho = params[1];
		} else {
			rho = params[0];
		}
	}

	@Override
	public void train() {
		int ny = Y.getColumnDimension();
		int maxIter = options.maxIter;
		if (Double.isInfinite(rho))
			rho = 1;
		double lambda = this.options.lambda;
		DenseVector weights = new DenseVector(n, 0);
		DenseVector GradWT = new DenseVector(n, 0);
		W = Matlab.ones(p, ny);
		Matrix GradW = W.copy();
		Matrix Theta = Matlab.ones(1, 1);
		Theta.setEntry(0, 0, rho);
		double gradTheta = 0;
		Matrix ThetaGrad = Matlab.ones(1, 1);
		Matrix XW = Matlab.zeros(Matlab.size(Y));
		
		double gval = 0;
		double hval = 0;
		
		mtimes(XW, X, W);
		gval = 0;
		for (int i = 0; i < n; i++) {
			double mu = XW.getEntry(i, 0);
			double t = T.getEntry(i, 0);
			double g = 1.0 / (1 + Math.exp(rho * (t - mu)));
			double y = Y.getEntry(i, 0);
			double e = g - y;
			gval += e * e;
			weights.set(i, (g - y) * rho * g * (1 - g));
		}
		operate(GradWT, weights, X);
		for (int j = 0; j < p; j++) {
			GradW.setEntry(j, 0, GradWT.get(j));
		}
		timesAssign(GradW, 2.0 / n);
		
		gval /= n;
		hval = lambda * norm(W, 1);
		
		ProximalMapping proxL1 = new ProxL1(lambda);
		ProximalMapping proxPlus = new ProxPlus();
		AcceleratedProximalGradient.type = 0;
		boolean flags[] = null;
		double epsilon = 1e-3;
		int k = 0;
		int APGMaxIter = 1000;
		
		double fval = 0;
		double fval_pre = 0;
		int cnt = 0;
		while(true) {
			
			// Update W:
			AcceleratedProximalGradient.prox = proxL1;
			while (true) {
				flags = AcceleratedProximalGradient.run(GradW, gval, hval, epsilon, W);
				// flags = NonnegativePLBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = LBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = AcceleratedGradientDescent.run(ThetaGrad, gval, epsilon, Theta);

				if (flags[0])
					break;

				gval = 0;
				mtimes(XW, X, W);
				for (int i = 0; i < n; i++) {
					double mu = XW.getEntry(i, 0);
					double t = T.getEntry(i, 0);
					double g = 1.0 / (1 + Math.exp(rho * (t - mu)));
					double y = Y.getEntry(i, 0);
					double e = g - y;
					gval += e * e;
					weights.set(i, (g - y) * rho * g * (1 - g));
				}
				
				gval /= n;
				hval = lambda * norm(W, 1);
				
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

					operate(GradWT, weights, X);
					for (int j = 0; j < p; j++) {
						GradW.setEntry(j, 0, GradWT.get(j));
					}
					timesAssign(GradW, 2.0 / n);
					
				}
				
			}
			
			// Update Theta:
			// We already computed gval for the new W, hval is always 0
			AcceleratedProximalGradient.prox = proxPlus;
			hval = 0;
			gradTheta = 0;
			for (int i = 0; i < n; i++) {
				double mu = XW.getEntry(i, 0);
				double t = T.getEntry(i, 0);
				double g = 1.0 / (1 + Math.exp(rho * (t - mu)));
				double y = Y.getEntry(i, 0);
				gradTheta += (g - y) * (mu - t) * g * (1 - g);
			}
			gradTheta *= 2.0 / n;
			ThetaGrad.setEntry(0, 0, gradTheta);
			while (true) {

				flags = AcceleratedProximalGradient.run(ThetaGrad, gval, hval, epsilon, Theta);
				// flags = NonnegativePLBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = LBFGS.run(ThetaGrad, gval, epsilon, Theta);
				// flags = AcceleratedGradientDescent.run(ThetaGrad, gval, epsilon, Theta);

				if (flags[0])
					break;
				
				gval = 0;
				gradTheta = 0;
				rho = Theta.getEntry(0, 0);
				for (int i = 0; i < n; i++) {
					double mu = XW.getEntry(i, 0);
					double t = T.getEntry(i, 0);
					double g = 1.0 / (1 + Math.exp(rho * (t - mu)));
					double y = Y.getEntry(i, 0);
					double e = g - y;
					gval += e * e;
					gradTheta += (g - y) * (mu - t) * g * (1 - g);
				}
				gval /= n;
				
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
					
					gradTheta *= 2.0 / n;
					ThetaGrad.setEntry(0, 0, gradTheta);
					
				}
				
			}

			cnt++;
			fval = gval + lambda * norm(W, 1);
			Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
			if ( cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps)
				//break;
			fval_pre = fval;
			if (cnt > maxIter)
				break;
			
		}
	}

	@Override
	public Matrix predict(Matrix Xt, Matrix Tt) {
		Matrix XtW = Xt.mtimes(W);
		int n = Xt.getRowDimension();
		Matrix PredY = new DenseMatrix(n, 1);
		for (int i = 0; i < n; i++) {
			double mu = XtW.getEntry(i, 0);
			double t = Tt.getEntry(i, 0);
			double g = 1.0 / (1 + Math.exp(rho * (t - mu)));
			PredY.setEntry(i, 0, g);
		}
		return PredY;
	}

	@Override
	public void loadModel(String filePath) {

		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			rho = ois.readDouble();
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
			oos.writeObject(new Double(rho));
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
