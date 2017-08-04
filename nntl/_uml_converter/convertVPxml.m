function [ ohfile, gr, oxmlfile, xsltfile, processedStyle ] = convertVPxml( infile, xsltfile )
grFileSfx = '_gr';
defaultXsltFile = fullfile(fileparts(mfilename),'layers2cpp.xsl');

if nargin<2
	if nargin<1
		ohfile = grFileSfx;
		gr = defaultXsltFile;
		return;
	end
	xsltfile = defaultXsltFile;
end

[p,fn] = fileparts(infile);
oxmlfile = fullfile(p, [fn grFileSfx '.xml']);
ohfile = fullfile(p, [fn '.h']);

gr=VPxml2graph(infile);

graph2xml(gr, oxmlfile);

[~,processedStyle] = xslt(oxmlfile, xsltfile, ohfile);

end

