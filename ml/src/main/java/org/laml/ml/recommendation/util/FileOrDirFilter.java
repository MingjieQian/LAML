package org.laml.ml.recommendation.util;

import java.io.File;
import java.io.FileFilter;

public class FileOrDirFilter implements FileFilter{
	String ext;
	public FileOrDirFilter(String ext) {
		this.ext = ext;
	}
	@Override
	public boolean accept(File pathname) {
		return pathname.isDirectory() || pathname.getName().endsWith(ext);
	}
}
