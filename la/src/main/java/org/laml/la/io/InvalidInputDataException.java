package la.io;

import java.io.File;

public class InvalidInputDataException extends Exception {

	private static final long serialVersionUID = 2945131732407207308L;

	private final int         _line;

	private File              _file;

	public InvalidInputDataException( String message, File file, int line ) {
		super(message);
		_file = file;
		_line = line;
	}
	
	public InvalidInputDataException( String message, int line ) {
		super(message);
		_file = null;
		_line = line;
	}

	public InvalidInputDataException( String message, String filename, int line ) {
		this(message, new File(filename), line);
	}

	public InvalidInputDataException( String message, File file, int lineNr, Exception cause ) {
		super(message, cause);
		_file = file;
		_line = lineNr;
	}
	
	public InvalidInputDataException( String message, int lineNr, Exception cause ) {
		super(message, cause);
		_file = null;
		_line = lineNr;
	}

	public InvalidInputDataException( String message, String filename, int lineNr, Exception cause ) {
		this(message, new File(filename), lineNr, cause);
	}

	public File getFile() {
		return _file;
	}

	public int getLine() {
		return _line;
	}

	@Override
	public String toString() {
		return super.toString() + " (" + _file + ":" + _line + ")";
	}

}
