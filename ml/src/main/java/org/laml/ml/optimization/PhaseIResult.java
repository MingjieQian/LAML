package org.laml.ml.optimization;

import org.laml.la.matrix.Matrix;

public class PhaseIResult {

	public boolean feasible;

	public Matrix optimizer;

	public double optimum;

	public PhaseIResult(Matrix optimizer, double optimum) {
		this.optimizer = optimizer;
		this.optimum = optimum;
	}

	public PhaseIResult(boolean feasible, Matrix optimizer, double optimum) {
		this.feasible = feasible;
		this.optimizer = optimizer;
		this.optimum = optimum;
	}

}
