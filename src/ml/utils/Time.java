package ml.utils;

public class Time {
	
	private static double t = 0;
	
	/**
	 * TSTART = TIC saves the time to an output argument, TSTART. 
	 * The numeric value of TSTART is only useful as an input 
	 * argument for a subsequent call to TOC.
	 * 
	 * @return time, in seconds, when TIC is called
	 * 
	 */
	public static double tic() {
		t = System.currentTimeMillis() / 1000d;
		return t;
	}
	
	/**
	 * Calculate the elapsed time, in seconds, since the most 
	 * recent execution of the TIC command.
	 * 
	 * @return elapsed time, in seconds, since the most recent 
	 *         execution of the TIC command
	 * 
	 */
	public static double toc() {
		return System.currentTimeMillis() / 1000d - t;
	}
	
	/**
	 * TOC(TSTART) measures the time elapsed since the TIC command that
     * generated TSTART.
	 * 
	 * @return elapsed time, in seconds, since the TIC command that
     *         generated TSTART
	 * 
	 */
	public static double toc(double TSTART) {
		return System.currentTimeMillis() / 1000d - TSTART;
	}
	
	/**
	 * PAUSE(n) pauses for n seconds before continuing, where n can also be a
     * fraction. The resolution of the clock is platform specific. Fractional
     * pauses of 0.01 seconds should be supported on most platforms.
     * 
	 * @param n time, in seconds, to pause
	 * 
	 */
	public static void pause(double n) {
		try {
			Thread.sleep((long)n * 1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

}
