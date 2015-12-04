package ml.regression;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.options.Options;

import static ml.utils.Matlab.*;

/***
 * Abstract super class for all regression methods.
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public abstract class Regression {

	/**
	 * Number of dependent variables.
	 */
	public int ny;
	
	/**
	 * Number of independent variables.
	 */
	public int p;
	
	/**
	 * Number of examples.
	 */
	public int n;
	
	/**
	 * Training data matrix (n x p) with each row being
	 * a data example.
	 */
	public Matrix X;
	
	/**
	 * Dependent variable matrix for training (n x ny).
	 */
	public Matrix Y;
	
	/**
	 * Unknown parameters represented as a matrix (p x ny).
	 */
	public Matrix W;
	
	/**
	 * Convergence tolerance.
	 */
	public double epsilon;
	
	/**
	 * Maximal number of iterations.
	 */
	public int maxIter;
	
	public Regression() {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		epsilon = 1e-6;
		maxIter = 600;
	}
	
	public Regression(double epsilon) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = epsilon;
		maxIter = 600;
	}
	
	public Regression(int maxIter, double epsilon) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = epsilon;
		this.maxIter = maxIter;
	}
	
	public Regression(Options options) {
		ny = 0;
		p = 0;
		n = 0;
		X = null;
		Y = null;
		W = null;
		this.epsilon = options.epsilon;
		this.maxIter = options.maxIter;
	}
	
	/**
	 * Feed training data for the regression model.
	 * 
	 * @param X data matrix with each row being a data example
	 */
	public void feedData(Matrix X) {
		this.X = X;
		p = X.getColumnDimension();
		n = X.getRowDimension();
		if (Y != null && X.getRowDimension() != Y.getRowDimension()) {
			System.err.println("The number of dependent variable vectors and " +
					"the number of data samples do not match!");
			System.exit(1);
		}
	}
	
	/**
	 * Feed training data for this regression model.
	 * 
	 * @param data an n x d 2D {@code double} array with each
	 *             row being a data example
	 */
	public void feedData(double[][] data) {
		feedData(new DenseMatrix(data));
	}
	
	/**
	 * Feed training dependent variables for this regression model.
	 * 
	 * @param Y dependent variable matrix for training with each row being
	 *          the dependent variable vector for each data training data
	 *          example
	 */
	public void feedDependentVariables(Matrix Y) {
		this.Y = Y;
		ny = Y.getColumnDimension();
		if (X != null && Y.getRowDimension() != n) {
			System.err.println("The number of dependent variable vectors and " +
					"the number of data samples do not match!");
			System.exit(1);
		}
	}
	
	/**
	 * Feed training dependent variables for this regression model.
	 * 
	 * @param depVars an n x c 2D {@code double} array
	 * 
	 */
	public void feedDependentVariables(double[][] depVars) {
		feedDependentVariables(new DenseMatrix(depVars));
	}
	
	/**
	 * Train the regression model.
	 */
	public abstract void train();
	
	/**
	 * Train the regression model given initial weight matrix W0.
	 * 
	 * @param W0 initial weight matrix (p x ny)
	 */
	public abstract void train(Matrix W0);
	
	/**
	 * Train the regression model given independent variable matrix
	 * X and dependent variable matrix Y.
	 * 
	 * @param X independent variable matrix (n x p)
	 * 
	 * @param Y dependent variable matrix (n x ny)
	 * 
	 * @return weight matrix (p x ny)
	 */
	public abstract Matrix train(Matrix X, Matrix Y);
	
	/**
	 * Train the regression model given independent variable matrix
	 * X, dependent variable matrix Y and the initial weight matrix
	 * W0.
	 * 
	 * @param X independent variable matrix (n x p)
	 * 
	 * @param Y dependent variable matrix (n x ny)
	 * 
	 * @param W0 initial weight matrix (p x ny)
	 * 
	 * @return weight matrix (p x ny)
	 */
	public abstract Matrix train(Matrix X, Matrix Y, Matrix W0);
	
	/**
	 * Predict the dependent variables for test data Xt.
	 * 
	 * @param Xt test data matrix with each row being a
	 *           data example.
	 *        
	 * @return dependent variables for Xt
	 * 
	 */
	public Matrix predict(Matrix Xt) {
		if (Xt.getColumnDimension() != p) {
			System.err.println("Dimensionality of the test data " +
					"doesn't match with the training data!");
			System.exit(1);
		}
		if (this instanceof LinearRegression) {
			return Xt.mtimes(W).plus(repmat(new DenseMatrix(((LinearRegression) this).B, 2), 3, 1));
		} else {
			return Xt.mtimes(W);
		}
	}
	
	/**
	 * Predict the dependent variables for test data Xt.
	 * 
	 * @param Xt an n x d 2D {@code double} array with each
	 *           row being a data example
	 *           
	 * @return dependent variables for Xt
	 * 
	 */
	public Matrix predict(double[][] Xt) {
		return predict(new DenseMatrix(Xt));
	}
	
	public abstract void loadModel(String filePath);
	
	public abstract void saveModel(String filePath);
	
}
