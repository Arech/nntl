function convertVPxmlFolder( fldr, xsltfile, bMoveDropoutToLph, bDropoutDefaultOn, dropoutDefaultType )
%CONVERTVPXMLFOLDER Summary of this function goes here
%   Detailed explanation goes here

[grFileSfx, defaultXsltFile] = convertVPxml();
grFileSfx = reverse(grFileSfx);
grFileSfxLen = length(grFileSfx);

if nargin<5
	dropoutDefaultType='DO';
	if nargin<4
		bDropoutDefaultOn=true;
		if nargin<3
			bMoveDropoutToLph = true;
			if nargin<2
				xsltfile = defaultXsltFile;
			end
		end
	end
end		
if isempty(xsltfile)
	xsltfile = defaultXsltFile;
end

assert(7==exist(fldr,'dir'), ['Specified folder `' fldr '` doesn`t exist!']);

dlist = dir([fldr '/*.xml']);
assert( ~isempty(dlist),'No xml files found');

processedStyle = xsltfile;

for ii=1:length(dlist)
	[~,fn] = fileparts(dlist(ii).name);		
	if dlist(ii).isdir || strncmpi(reverse(fn), grFileSfx, grFileSfxLen)
		continue;
	end
	
	fprintf(1,'\t%s :\n', dlist(ii).name);
	[ ~, ~, ~, ~, processedStyle ] = convertVPxml( fullfile(dlist(ii).folder, dlist(ii).name)...
		, processedStyle, bMoveDropoutToLph, bDropoutDefaultOn, dropoutDefaultType);	
end

end

