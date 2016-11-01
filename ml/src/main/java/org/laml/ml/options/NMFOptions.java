package org.laml.ml.options;

public class NMFOptions extends ClusteringOptions {

	public double epsilon;
	public boolean calc_OV;
	
	public NMFOptions() {
		super();
		epsilon = 1e-6;
		calc_OV = false;
	}
	
	public NMFOptions(NMFOptions NMFOptions) {
		super((ClusteringOptions)NMFOptions);
		epsilon = NMFOptions.epsilon;
		calc_OV = NMFOptions.calc_OV;
	}

	public NMFOptions(int nClus) {
		super(nClus);
		epsilon = 1e-6;
		calc_OV = false;
	}

	public NMFOptions(int nClus, boolean verbose, int maxIter) {
		super(nClus, verbose, maxIter);
		epsilon = 1e-6;
		calc_OV = false;
	}

	public NMFOptions(ClusteringOptions options) {
		super(options);
		epsilon = 1e-6;
		calc_OV = false;
	}

}
