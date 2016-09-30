package ml.options;

public class Options {
	
	public boolean verbose;
	public int nFeature;
	public int nClass;
	public int nTopic;
	public int nTerm;
	public int nDoc;
	public int nTopTerm;
	public double epsilon;
	public int maxIter;
	
	public double gamma;
	public double mu;
	public double lambda;
	
	public boolean calc_OV;
	
	public int nClus;
	
	public Options(Options o) {
		
		verbose = o.verbose;
		nFeature = o.nFeature;
		nClass = o.nClass;
		nTopic = o.nTopic;
		nTerm = o.nTerm;		
		nDoc = o.nDoc;
		nClus = o.nClus;
		
		nTopTerm = o.nTopTerm;
		
		epsilon = o.epsilon;
		maxIter = o.maxIter;
		gamma = o.gamma;
		mu = o.mu;
		calc_OV = o.calc_OV;
		
		lambda = o.lambda;
		
	}

	public Options() {
		
		verbose = false;
		nFeature = 1;
		nClass = 1;
		nTopic = 1;
		nTerm = 1;		
		nDoc = 0;
		
		nTopTerm = nTerm;
		
		epsilon = 1e-6;
		maxIter = 300;
		gamma = 0.0001;
		mu = 0.1;
		calc_OV = false;
		
		lambda = 1.0;
		
		nClus = 1;
		
	}
	
}
