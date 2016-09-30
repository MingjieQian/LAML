package ml.options;

public class L1NMFOptions extends ClusteringOptions {

	public double gamma;
	public double mu;
	public double epsilon;
	public boolean calc_OV;
	
	public L1NMFOptions() {
		super();
		gamma = 0.0001;
		mu = 0.1;
		epsilon = 1e-6;
		calc_OV = false;
	}
	
	public L1NMFOptions(L1NMFOptions L1NMFOptions) {
		super((ClusteringOptions)L1NMFOptions);
		gamma = L1NMFOptions.gamma;
		mu = L1NMFOptions.mu;
		epsilon = L1NMFOptions.epsilon;
		calc_OV = L1NMFOptions.calc_OV;
	}

	public L1NMFOptions(int nClus) {
		super(nClus);
		gamma = 0.0001;
		mu = 0.1;
		epsilon = 1e-6;
		calc_OV = false;
	}

	public L1NMFOptions(int nClus, boolean verbose, int maxIter) {
		super(nClus, verbose, maxIter);
		gamma = 0.0001;
		mu = 0.1;
		epsilon = 1e-6;
		calc_OV = false;
	}

	public L1NMFOptions(ClusteringOptions options) {
		super(options);
		gamma = 0.0001;
		mu = 0.1;
		epsilon = 1e-6;
		calc_OV = false;
	}

}
