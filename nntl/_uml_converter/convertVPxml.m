function [ ohfile, gr, oxmlfile, xsltfile, processedStyle ] = convertVPxml( infile, xsltfile, bMoveDropoutToLph, bDropoutDefaultOn, dropoutDefaultType)
grFileSfx = '_gr';
defaultXsltFile = fullfile(fileparts(mfilename),'layers2cpp.xsl');

if nargin<5
	dropoutDefaultType='DO';
	if nargin<4
		bDropoutDefaultOn=true;
		if nargin<3
			bMoveDropoutToLph = true;
			if nargin<2
				if nargin<1
					ohfile = grFileSfx;
					gr = defaultXsltFile;
					return;
				end
				xsltfile = defaultXsltFile;
			end
		end
	end
end
if isempty(xsltfile)
	xsltfile = defaultXsltFile;
end

[p,fn] = fileparts(infile);
oxmlfile = fullfile(p, [fn grFileSfx '.xml']);
ohfile = fullfile(p, [fn '.h']);

gr=VPxml2graph(infile, bMoveDropoutToLph, bDropoutDefaultOn, dropoutDefaultType);

graph2xml(gr, oxmlfile);

[~,processedStyle] = xslt(oxmlfile, xsltfile, ohfile);

end

