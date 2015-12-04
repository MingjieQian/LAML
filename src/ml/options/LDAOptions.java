package ml.options;

public class LDAOptions {
	
	public boolean hasLabel;
	public boolean verbose;
	public int nTopic;
	public int nTerm;
	public double alpha;
	public double beta;
	public int iterations;
	public int burnIn;
	public int thinInterval;
	public int sampleLag;
	

	public LDAOptions() {
		
		hasLabel = false;
		verbose = false;
		
		nTerm = 1;		
		nTopic = 1;
		
		alpha = 50.0 / nTopic;
		beta = 200.0 / nTerm;
		
		iterations = 1000;
		burnIn = 500;
		thinInterval = 50;
        sampleLag = 10;
	}
	
}
